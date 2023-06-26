<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-llm: Integrating LLMs into structured NLP pipelines

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/explosion/spacy-llm/test.yml?branch=main)](https://github.com/explosion/spacy-llm/actions/workflows/test.yml)
[![pypi Version](https://img.shields.io/pypi/v/spacy-llm.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spacy-llm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

This package integrates Large Language Models (LLMs) into [spaCy](https://spacy.io), featuring a modular system for **fast prototyping** and **prompting**, and turning unstructured responses into **robust outputs** for various NLP tasks, **no training data** required.

- Serializable `llm` **component** to integrate prompts into your pipeline
- **Modular functions** to define the [**task**](#tasks) (prompting and parsing) and [**backend**](#backends) (model to use)
- Support for **hosted APIs** and self-hosted **open-source models**
- Integration with [`MiniChain`](https://github.com/srush/MiniChain) and [`LangChain`](https://github.com/hwchase17/langchain)
- Access to **[OpenAI API](https://platform.openai.com/docs/api-reference/introduction)**, including GPT-4 and various GPT-3 models
- Built-in support for **open-source [Dolly](https://huggingface.co/databricks)** models hosted on Hugging Face
- Usage examples for **Named Entity Recognition** and **Text Classification**
- Easy implementation of **your own functions** via [spaCy's registry](https://spacy.io/api/top-level#registry) for custom prompting, parsing and model integrations

## 🧠 Motivation

Large Language Models (LLMs) feature powerful natural language understanding capabilities. With only a few (and sometimes no) examples, an LLM can be prompted to perform custom NLP tasks such as text categorization, named entity recognition, coreference resolution, information extraction and more.

[spaCy](https://spacy.io) is a well-established library for building systems that need to work with language in various ways. spaCy's built-in components are generally powered by supervised learning or rule-based approaches.

Supervised learning is much worse than LLM prompting for prototyping, but for many tasks it's much better for production. A transformer model that runs comfortably on a single GPU is extremely powerful, and it's likely to be a better choice for any task for which you have a well-defined output. You train the model with anything from a few hundred to a few thousand labelled examples, and it will learn to do exactly that. Efficiency, reliability and control are all better with supervised learning, and accuracy will generally be higher than LLM prompting as well.

`spacy-llm` lets you have **the best of both worlds**. You can quickly initialize a pipeline with components powered by LLM prompts, and freely mix in components powered by other approaches. As your project progresses, you can look at replacing some or all of the LLM-powered components as you require.

Of course, there can be components in your system for which the power of an LLM is fully justified. If you want a system that can synthesize information from multiple documents in subtle ways and generate a nuanced summary for you, bigger is better. However, even if your production system needs an LLM for some of the task, that doesn't mean you need an LLM for all of it. Maybe you want to use a cheap text classification model to help you find the texts to summarize, or maybe you want to add a rule-based system to sanity check the output of the summary. These before-and-after tasks are much easier with a mature and well-thought-out library, which is exactly what spaCy provides.

## ⏳ Install

`spacy-llm` will be installed automatically in future spaCy versions. For now, you can run the following in the same virtual environment where you already have `spacy` [installed](https://spacy.io/usage).

```bash
python -m pip install spacy-llm
```

> ⚠️ This package is still experimental and it is possible that changes made to the interface will be breaking in minor version updates.

## 🐍 Usage

The task and the backend have to be supplied to the `llm` pipeline component using [spaCy's config
system](https://spacy.io/api/data-formats#config). This package provides various built-in
functionality, as detailed in the [API](#-api) documentation.

### Example 1: Add a text classifier using a GPT-3 model from OpenAI

Create a new API key from openai.com or fetch an existing one, and ensure the keys are set as environmental variables.
For more background information, see the [OpenAI](#openai) section.

Create a config file `config.cfg` containing at least the following
(or see the full example [here](usage_examples/textcat_openai)):

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = ["COMPLIMENT", "INSULT"]

[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "gpt-3.5-turbo", "temperature": 0.3}
```

Now run:

```python
from spacy_llm.util import assemble

nlp = assemble("config.cfg")
doc = nlp("You look gorgeous!")
print(doc.cats)
```

### Example 2: Add NER using an open-source model through Hugging Face

To run this example, ensure that you have a GPU enabled, and `transformers`, `torch` and CUDA installed.
For more background information, see the [DollyHF](#spacydollyhfv1) section.

Create a config file `config.cfg` containing at least the following
(or see the full example [here](usage_examples/ner_dolly)):

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = ["PERSON", "ORGANISATION", "LOCATION"]

[components.llm.backend]
@llm_backends = "spacy.Dolly_HF.v1"
# For better performance, use databricks/dolly-v2-12b instead
model = "databricks/dolly-v2-3b"
```

Now run:

```python
from spacy_llm.util import assemble

nlp = assemble("config.cfg")
doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
print([(ent.text, ent.label_) for ent in doc.ents])
```

Note that Hugging Face will download the `"databricks/dolly-v2-3b"` model the first time you use it. You can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.
Also, you can upgrade the model to be `"databricks/dolly-v2-12b"` for better performance.

### Example 3: Create the component directly in Python

The `llm` component behaves as any other spaCy component does, so adding it to an existing pipeline follows the same
pattern:

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe(
    "llm",
    config={
        "task": {
            "@llm_tasks": "spacy.NER.v2",
            "labels": ["PERSON", "ORGANISATION", "LOCATION"]
        },
        "backend": {
            "@llm_backends": "spacy.REST.v1",
            "api": "OpenAI",
            "config": {"model": "gpt-3.5-turbo"},
        },
    },
)
nlp.initialize()
doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
print([(ent.text, ent.label_) for ent in doc.ents])
```

Note that for efficient usage of resources, typically you would use [`nlp.pipe(docs)`](https://spacy.io/api/language#pipe)
with a batch, instead of calling `nlp(doc)` with a single document.

### Example 4: Implement your own custom task

To write a
[`task`](#tasks), you
need to implement two functions: `generate_prompts` that takes a list of spaCy [`Doc`](https://spacy.io/api/doc) objects and transforms
them into a list of prompts, and `parse_responses` that transforms the LLM outputs into annotations on the [`Doc`](https://spacy.io/api/doc), e.g. entity spans, text categories and more.

To register your custom task with spaCy, decorate a factory function using the `spacy_llm.registry.llm_tasks` decorator with a custom name that you can refer to in your config.

> 📖 For more details, see the [**usage example on writing your own task**](usage_examples/README.md#writing-your-own-task)

```python
from typing import Iterable, List
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.util import split_labels


@registry.llm_tasks("my_namespace.MyTask.v1")
def make_my_task(labels: str, my_other_config_val: float) -> "MyTask":
    labels_list = split_labels(labels)
    return MyTask(labels=labels_list, my_other_config_val=my_other_config_val)


class MyTask:
    def __init__(self, labels: List[str], my_other_config_val: float):
        ...

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        ...
```

```ini
# config.cfg (excerpt)
[components.llm.task]
@llm_tasks = "my_namespace.MyTask.v1"
labels = LABEL1,LABEL2,LABEL3
my_other_config_val = 0.3
```

## Logging

spacy-llm has a built-in logger that can log the prompt sent to the LLM as well as its raw response. This logger uses the debug level and by default has a `logging.NullHandler()` configured.

In order to use this logger, you can setup a simple handler like this:

```python
import logging
import spacy_llm


spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.DEBUG)
```

> NOTE: Any `logging` handler will work here so you probably want to use some sort of rotating `FileHandler` as the generated prompts can be quite long, especially for tasks with few-shot examples.

Then when using the pipeline you'll be able to view the prompt and response.

E.g. with the config and code from [Example 1](##example-1-add-a-text-classifier-using-a-gpt-3-model-from-openai) above:

```python
from spacy_llm.util import assemble


nlp = assemble("config.cfg")
doc = nlp("You look gorgeous!")
print(doc.cats)
```

You will see `logging` output similar to:

```
Generated prompt for doc: You look gorgeous!

You are an expert Text Classification system. Your task is to accept Text as input
and provide a category for the text based on the predefined labels.

Classify the text below to any of the following labels: COMPLIMENT, INSULT
The task is non-exclusive, so you can provide more than one label as long as
they're comma-delimited. For example: Label1, Label2, Label3.
Do not put any other text in your answer, only one or more of the provided labels with nothing before or after.
If the text cannot be classified into any of the provided labels, answer `==NONE==`.

Here is the text that needs classification


Text:
'''
You look gorgeous!
'''

Backend response for doc: You look gorgeous!
COMPLIMENT
```

`print(doc.cats)` to standard output should look like:

```
{'COMPLIMENT': 1.0, 'INSULT': 0.0}
```

## 📓 API

`spacy-llm` exposes a `llm` factory that accepts the following configuration options:

| Argument         | Type                                        | Description                                                                         |
| ---------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| `task`           | `Optional[LLMTask]`                         | An LLMTask can generate prompts and parse LLM responses. See [docs](#tasks).        |
| `backend`        | `Callable[[Iterable[Any]], Iterable[Any]]]` | Callable querying a specific LLM API. See [docs](#backends).                        |
| `cache`          | `Cache`                                     | Cache to use for caching prompts and responses per doc (batch). See [docs](#cache). |
| `save_io`        | `bool`                                      | Whether to save prompts/responses within `Doc.user_data["llm_io"]`.                 |
| `validate_types` | `bool`                                      | Whether to check if signatures of configured backend and task are consistent.       |

An `llm` component is defined by two main settings:

- A [**task**](#tasks), defining the prompt to send to the LLM as well as the functionality to parse the resulting response
  back into structured fields on spaCy's [Doc](https://spacy.io/api/doc) objects.
- A [**backend**](#backends) defining the model to use and how to connect to it. Note that `spacy-llm` supports both access to external
  APIs (such as OpenAI) as well as access to self-hosted open-source LLMs (such as using Dolly through Hugging Face).

Moreover, `spacy-llm` exposes a customizable [**caching**](#cache) functionality to avoid running
the same document through an LLM service (be it local or through a REST API) more than once.

Finally, you can choose to save a stringified version of LLM prompts/responses
within the `Doc.user_data["llm_io"]` attribute by setting `save_io` to `True`.
`Doc.user_data["llm_io"]` is a dictionary containing one entry for every LLM component
within the spaCy pipeline. Each entry is itself a dictionary, with two keys:
`prompt` and `response`.

A note on `validate_types`: by default, `spacy-llm` checks whether the signatures of the `backend` and `task` callables
are consistent with each other and emits a warning if they don't. `validate_types` can be set to `False` if you want to
disable this behavior.

### Tasks

A _task_ defines an NLP problem or question, that will be sent to the LLM via a prompt. Further, the task defines
how to parse the LLM's responses back into structured information. All tasks are registered in spaCy's `llm_tasks` registry.

Practically speaking, a task should adhere to the `Protocol` `LLMTask` defined in [`ty.py`](spacy_llm/ty.py).
It needs to define a `generate_prompts` function and a `parse_responses` function.

Moreover, the task may define an optional [`scorer` method](https://spacy.io/api/scorer#score).
It should accept an iterable of `Example`s as input and return a score dictionary.
If the `scorer` method is defined, `spacy-llm` will call it to evaluate the component.

#### <kbd>function</kbd> `task.generate_prompts`

Takes a collection of documents, and returns a collection of "prompts", which can be of type `Any`.
Often, prompts are of type `str` - but this is not enforced to allow for maximum flexibility in the framework.

| Argument    | Type            | Description            |
| ----------- | --------------- | ---------------------- |
| `docs`      | `Iterable[Doc]` | The input documents.   |
| **RETURNS** | `Iterable[Any]` | The generated prompts. |

#### <kbd>function</kbd> `task.parse_responses`

Takes a collection of LLM responses and the original documents, parses the responses into structured information,
and sets the annotations on the documents. The `parse_responses` function is free to set the annotations in any way,
including `Doc` fields like `ents`, `spans` or `cats`, or using custom defined fields.

The `responses` are of type `Iterable[Any]`, though they will often be `str` objects. This depends on the
return type of the [backend](#backends).

| Argument    | Type            | Description              |
| ----------- | --------------- | ------------------------ |
| `docs`      | `Iterable[Doc]` | The input documents.     |
| `responses` | `Iterable[Any]` | The generated prompts.   |
| **RETURNS** | `Iterable[Doc]` | The annotated documents. |

#### spacy.NER.v2

The built-in NER task supports both zero-shot and few-shot prompting. This version also supports explicitly defining the provided labels with custom descriptions.

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = ["PERSON", "ORGANISATION", "LOCATION"]
examples = null
```

| Argument                  | Type                                    | Default                                                  | Description                                                                                                                                           |
| ------------------------- | --------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | `Union[List[str], str]`                 |                                                          | List of labels or str of comma-separated list of labels.                                                                                              |
| `template`                | `str`                                   | [ner.v2.jinja](./spacy_llm/tasks/templates/ner.v2.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.              |
| `label_definitions`       | `Optional[Dict[str, str]]`              | `None`                                                   | Optional dict mapping a label to a description of that label. These descriptions are added to the prompt to help instruct the LLM on what to extract. |
| `examples`                | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                   | Optional function that generates examples for few-shot learning.                                                                                      |
| `normalizer`              | `Optional[Callable[[str], str]]`        | `None`                                                   | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                                    |
| `alignment_mode`          | `str`                                   | `"contract"`                                             | Alignment mode in case the LLM returns entities that do not align with token boundaries. Options are `"strict"`, `"contract"` or `"expand"`.          |
| `case_sensitive_matching` | `bool`                                  | `False`                                                  | Whether to search without case sensitivity.                                                                                                           |
| `single_match`            | `bool`                                  | `False`                                                  | Whether to match an entity in the LLM's response only once (the first hit) or multiple times.                                                         |

The NER task implementation doesn't currently ask the LLM for specific offsets, but simply expects a list of strings that represent the enties in the document.
This means that a form of string matching is required. This can be configured by the following parameters:

- The `single_match` parameter is typically set to `False` to allow for multiple matches. For instance, the response from the LLM might only mention the entity "Paris" once, but you'd still
  want to mark it every time it occurs in the document.
- The case-sensitive matching is typically set to `False` to be robust against case variances in the LLM's output.
- The `alignment_mode` argument is used to match entities as returned by the LLM to the tokens from the original `Doc` - specifically it's used as argument
  in the call to [`doc.char_span()`](https://spacy.io/api/doc#char_span). The `"strict"` mode will only keep spans that strictly adhere to the given token boundaries.
  `"contract"` will only keep those tokens that are fully within the given range, e.g. reducing `"New Y"` to `"New"`.
  Finally, `"expand"` will expand the span to the next token boundaries, e.g. expanding `"New Y"` out to `"New York"`.

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```yaml
- text: Jack and Jill went up the hill.
  entities:
    PERSON:
      - Jack
      - Jill
    LOCATION:
      - hill
- text: Jack fell down and broke his crown.
  entities:
    PERSON:
      - Jack
```

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = PERSON,ORGANISATION,LOCATION
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

If you don't have specific examples to provide to the LLM, you can write definitions for each label and provide them via the `label_definitions` argument. This lets you tell the LLM exactly what you're looking for rather than relying on the LLM to interpret its task given just the label name. Label descriptions are freeform so you can write whatever you want here, but through some experiments a brief description along with some examples and counter examples seems to work quite well.

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = PERSON,SPORTS_TEAM
[components.llm.task.label_definitions]
PERSON = "Extract any named individual in the text."
SPORTS_TEAM = "Extract the names of any professional sports team. e.g. Golden State Warriors, LA Lakers, Man City, Real Madrid"
```

> Label descriptions can also be used with explicit examples to give as much info to the LLM backend as possible.

#### spacy.NER.v1

The original version of the built-in NER task supports both zero-shot and few-shot prompting.

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION
examples = null
```

| Argument                  | Type                                    | Default      | Description                                                                                                                                  |
| ------------------------- | --------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | `str`                                   |              | Comma-separated list of labels.                                                                                                              |
| `examples`                | `Optional[Callable[[], Iterable[Any]]]` | `None`       | Optional function that generates examples for few-shot learning.                                                                             |
| `normalizer`              | `Optional[Callable[[str], str]]`        | `None`       | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                           |
| `alignment_mode`          | `str`                                   | `"contract"` | Alignment mode in case the LLM returns entities that do not align with token boundaries. Options are `"strict"`, `"contract"` or `"expand"`. |
| `case_sensitive_matching` | `bool`                                  | `False`      | Whether to search without case sensitivity.                                                                                                  |
| `single_match`            | `bool`                                  | `False`      | Whether to match an entity in the LLM's response only once (the first hit) or multiple times.                                                |

The NER task implementation doesn't currently ask the LLM for specific offsets, but simply expects a list of strings that represent the enties in the document.
This means that a form of string matching is required. This can be configured by the following parameters:

- The `single_match` parameter is typically set to `False` to allow for multiple matches. For instance, the response from the LLM might only mention the entity "Paris" once, but you'd still
  want to mark it every time it occurs in the document.
- The case-sensitive matching is typically set to `False` to be robust against case variances in the LLM's output.
- The `alignment_mode` argument is used to match entities as returned by the LLM to the tokens from the original `Doc` - specifically it's used as argument
  in the call to [`doc.char_span()`](https://spacy.io/api/doc#char_span). The `"strict"` mode will only keep spans that strictly adhere to the given token boundaries.
  `"contract"` will only keep those tokens that are fully within the given range, e.g. reducing `"New Y"` to `"New"`.
  Finally, `"expand"` will expand the span to the next token boundaries, e.g. expanding `"New Y"` out to `"New York"`.

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```yaml
- text: Jack and Jill went up the hill.
  entities:
    PERSON:
      - Jack
      - Jill
    LOCATION:
      - hill
- text: Jack fell down and broke his crown.
  entities:
    PERSON:
      - Jack
```

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

#### spacy.SpanCat.v2

The built-in SpanCat task is a simple adaptation of the NER task to
support overlapping entities and store its annotations in `doc.spans`.

```ini
[components.llm.task]
@llm_tasks = "spacy.SpanCat.v2"
labels = ["PERSON", "ORGANISATION", "LOCATION"]
examples = null
```

| Argument                  | Type                                    | Default                                                            | Description                                                                                                                                           |
| ------------------------- | --------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | `Union[List[str], str]`                 |                                                                    | List of labels or str of comma-separated list of labels.                                                                                              |
| `template`                | `str`                                   | [`spancat.v2.jinja`](./spacy_llm/tasks/templates/spancat.v2.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.              |
| `label_definitions`       | `Optional[Dict[str, str]]`              | `None`                                                             | Optional dict mapping a label to a description of that label. These descriptions are added to the prompt to help instruct the LLM on what to extract. |
| `spans_key`               | `str`                                   | `"sc"`                                                             | Key of the `Doc.spans` dict to save the spans under.                                                                                                  |
| `examples`                | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                             | Optional function that generates examples for few-shot learning.                                                                                      |
| `normalizer`              | `Optional[Callable[[str], str]]`        | `None`                                                             | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                                    |
| `alignment_mode`          | `str`                                   | `"contract"`                                                       | Alignment mode in case the LLM returns entities that do not align with token boundaries. Options are `"strict"`, `"contract"` or `"expand"`.          |
| `case_sensitive_matching` | `bool`                                  | `False`                                                            | Whether to search without case sensitivity.                                                                                                           |
| `single_match`            | `bool`                                  | `False`                                                            | Whether to match an entity in the LLM's response only once (the first hit) or multiple times.                                                         |

Except for the `spans_key` parameter, the SpanCat task reuses the configuration
from the NER task.
Refer to [its documentation](#spacynerv2) for more insight.

#### spacy.SpanCat.v1

The original version of the built-in SpanCat task is a simple adaptation of the v1 NER task to
support overlapping entities and store its annotations in `doc.spans`.

```ini
[components.llm.task]
@llm_tasks = "spacy.SpanCat.v1"
labels = PERSON,ORGANISATION,LOCATION
examples = null
```

| Argument                  | Type                                    | Default      | Description                                                                                                                                  |
| ------------------------- | --------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | `str`                                   |              | Comma-separated list of labels.                                                                                                              |
| `spans_key`               | `str`                                   | `"sc"`       | Key of the `Doc.spans` dict to save the spans under.                                                                                         |
| `examples`                | `Optional[Callable[[], Iterable[Any]]]` | `None`       | Optional function that generates examples for few-shot learning.                                                                             |
| `normalizer`              | `Optional[Callable[[str], str]]`        | `None`       | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                           |
| `alignment_mode`          | `str`                                   | `"contract"` | Alignment mode in case the LLM returns entities that do not align with token boundaries. Options are `"strict"`, `"contract"` or `"expand"`. |
| `case_sensitive_matching` | `bool`                                  | `False`      | Whether to search without case sensitivity.                                                                                                  |
| `single_match`            | `bool`                                  | `False`      | Whether to match an entity in the LLM's response only once (the first hit) or multiple times.                                                |

Except for the `spans_key` parameter, the SpanCat task reuses the configuration
from the NER task.
Refer to [its documentation](#spacynerv1) for more insight.

#### spacy.TextCat.v2

The built-in TextCat task supports both zero-shot and few-shot prompting.

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = ["COMPLIMENT", "INSULT"]
examples = null
```

| Argument            | Type                                    | Default                                                      | Description                                                                                                                                      |
| ------------------- | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `labels`            | `Union[List[str], str]`                 |                                                              | List of labels or str of comma-separated list of labels.                                                                                         |
| `template`          | `str`                                   | [`textcat.jinja`](./spacy_llm/tasks/templates/textcat.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.         |
| `examples`          | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                       | Optional function that generates examples for few-shot learning.                                                                                 |
| `normalizer`        | `Optional[Callable[[str], str]]`        | `None`                                                       | Function that normalizes the labels as returned by the LLM. If `None`, falls back to `spacy.LowercaseNormalizer.v1`.                             |
| `exclusive_classes` | `bool`                                  | `False`                                                      | If set to `True`, only one label per document should be valid. If set to `False`, one document can have multiple labels.                         |
| `allow_none`        | `bool`                                  | `True`                                                       | When set to `True`, allows the LLM to not return any of the given label. The resulting dict in `doc.cats` will have `0.0` scores for all labels. |
| `verbose`           | `bool`                                  | `False`                                                      | If set to `True`, warnings will be generated when the LLM returns invalid responses.                                                             |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```json
[
  {
    "text": "You look great!",
    "answer": "Compliment"
  },
  {
    "text": "You are not very clever at all.",
    "answer": "Insult"
  }
]
```

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = ["COMPLIMENT", "INSULT"]
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "textcat_examples.json"
```

#### spacy.TextCat.v1

The original version of the built-in TextCat task supports both zero-shot and few-shot prompting.

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = COMPLIMENT,INSULT
examples = null
```

| Argument            | Type                                    | Default | Description                                                                                                                                      |
| ------------------- | --------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `labels`            | str                                     |         | Comma-separated list of labels.                                                                                                                  |
| `examples`          | `Optional[Callable[[], Iterable[Any]]]` | `None`  | Optional function that generates examples for few-shot learning.                                                                                 |
| `normalizer`        | `Optional[Callable[[str], str]]`        | `None`  | Function that normalizes the labels as returned by the LLM. If `None`, falls back to `spacy.LowercaseNormalizer.v1`.                             |
| `exclusive_classes` | `bool`                                  | `False` | If set to `True`, only one label per document should be valid. If set to `False`, one document can have multiple labels.                         |
| `allow_none`        | `bool`                                  | `True`  | When set to `True`, allows the LLM to not return any of the given label. The resulting dict in `doc.cats` will have `0.0` scores for all labels. |
| `verbose`           | `bool`                                  | `False` | If set to `True`, warnings will be generated when the LLM returns invalid responses.                                                             |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```json
[
  {
    "text": "You look great!",
    "answer": "Compliment"
  },
  {
    "text": "You are not very clever at all.",
    "answer": "Insult"
  }
]
```

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = COMPLIMENT,INSULT
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "textcat_examples.json"
```

#### spacy.REL.v1

The built-in REL task supports both zero-shot and few-shot prompting.
It relies on an upstream NER component for entities extraction.

```ini
[components.llm.task]
@llm_tasks = "spacy.REL.v1"
labels = ["LivesIn", "Visits"]
```

| Argument            | Type                                    | Default                                              | Description                                                                                                                              |
| ------------------- | --------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`            | `Union[List[str], str]`                 |                                                      | List of labels or str of comma-separated list of labels.                                                                                 |
| `template`          | `str`                                   | [`rel.jinja`](./spacy_llm/tasks/templates/rel.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `label_description` | `Optional[Dict[str, str]]`              | `None`                                               | Dictionary providing a description for each relation label.                                                                              |
| `examples`          | `Optional[Callable[[], Iterable[Any]]]` | `None`                                               | Optional function that generates examples for few-shot learning.                                                                         |
| `normalizer`        | `Optional[Callable[[str], str]]`        | `None`                                               | Function that normalizes the labels as returned by the LLM. If `None`, falls back to `spacy.LowercaseNormalizer.v1`.                     |
| `verbose`           | `bool`                                  | `False`                                              | If set to `True`, warnings will be generated when the LLM returns invalid responses.                                                     |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```json
{"text": "Laura bought a house in Boston with her husband Mark.", "ents": [{"start_char": 0, "end_char": 5, "label": "PERSON"}, {"start_char": 24, "end_char": 30, "label": "GPE"}, {"start_char": 48, "end_char": 52, "label": "PERSON"}], "relations": [{"dep": 0, "dest": 1, "relation": "LivesIn"}, {"dep": 2, "dest": 1, "relation": "LivesIn"}]}
{"text": "Michael travelled through South America by bike.", "ents": [{"start_char": 0, "end_char": 7, "label": "PERSON"}, {"start_char": 26, "end_char": 39, "label": "LOC"}], "relations": [{"dep": 0, "dest": 1, "relation": "Visits"}]}
```

Note: the REL task relies on pre-extracted entities to make its prediction.
Hence, you'll need to add a component that populates `doc.ents` with recognized
spans to your spaCy pipeline and put it _before_ the REL component.

```ini
[components.llm.task]
@llm_tasks = "spacy.REL.v1"
labels = ["LivesIn", "Visits"]
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "rel_examples.jsonl"
```

#### spacy.Lemma.v1

The `Lemma.v1` task lemmatizes the provided text and updates the `lemma_` attribute in the doc's tokens accordingly.

```ini
[components.llm.task]
@llm_tasks = "spacy.Lemma.v1"
examples = null
```

| Argument   | Type                                    | Default                                                | Description                                                                                                                              |
| ---------- | --------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `template` | `str`                                   | [lemma.jinja](./spacy_llm/tasks/templates/lemma.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `examples` | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                 | Optional function that generates examples for few-shot learning.                                                                         |

`Lemma.v1` prompts the LLM to lemmatize the passed text and return the lemmatized version as a list of tokens and their
corresponding lemma. E. g. the text
`I'm buying ice cream for my friends` should invoke the response

```
I: I
'm: be
buying: buy
ice: ice
cream: cream
for: for
my: my
friends: friend
.: .
```

If for any given text/doc instance the number of lemmas returned by the LLM doesn't match the number of tokens recognized
by spaCy, no lemmas are stored in the corresponding doc's tokens. Otherwise the tokens `.lemma_` property is updated with
the lemma suggested by the LLM.

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```yaml
- text: I'm buying ice cream.
  lemmas:
    - "I": "I"
    - "'m": "be"
    - "buying": "buy"
    - "ice": "ice"
    - "cream": "cream"
    - ".": "."

- text: I've watered the plants.
  lemmas:
    - "I": "I"
    - "'ve": "have"
    - "watered": "water"
    - "the": "the"
    - "plants": "plant"
    - ".": "."
```

```ini
[components.llm.task]
@llm_tasks = "spacy.Lemma.v1"
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "lemma_examples.yml"
```

#### spacy.NoOp.v1

This task is only useful for testing - it tells the LLM to do nothing, and does not set any fields on the `docs`.

```ini
[components.llm.task]
@llm_tasks = "spacy.NoOp.v1"
```

### Backends

A _backend_ defines which LLM model to query, and how to query it. It can be a simple function taking a collection
of prompts (consistent with the output type of `task.generate_prompts()`) and returning a collection of responses
(consistent with the expected input of `parse_responses`). Generally speaking, it's a function of type `Callable[[Iterable[Any]], Iterable[Any]]`,
but specific implementations can have other signatures, like `Callable[[Iterable[str]], Iterable[str]]`.

All built-in backends are registered in `llm_backends`. If no backend is specified, the repo currently connects to the [`OpenAI` API](#openai) by default,
using the built-in REST protocol, and accesses the `"gpt-3.5-turbo"` model.

> :question: _Why are there backends for third-party libraries in addition to a native REST backend and which should
> I choose?_
>
> Third-party libraries like `langchain` or `minichain` focus on prompt management, integration of many different LLM
> APIs, and other related features such as conversational memory or agents. `spacy-llm` on the other hand emphasizes
> features we consider useful in the context of NLP pipelines utilizing LLMs to process documents (mostly) independent
> from each other. It makes sense that the feature set of such third-party libraries and `spacy-llm` is not identical -
> and users might want to take advantage of features not available in `spacy-llm`.
>
> The advantage of offering our own REST backend is that we can ensure a larger degree of stability of robustness, as
> we can guarantee backwards-compatibility and more smoothly integrated error handling.
>
> Ultimately we recommend trying to implement your use case using the REST backend first (which is configured as the
> default backend). If however there are features or APIs not covered by `spacy-llm`, it's trivial to switch to the
> backend of a third-party library - and easy to customize the prompting mechanism, if so required.

#### OpenAI

When the backend uses OpenAI, you have to get an API key from openai.com, and ensure that the keys are set as
environmental variables:

```shell
export OPENAI_API_KEY="sk-..."
export OPENAI_API_ORG="org-..."
```

#### spacy.REST.v1

This default backend uses `requests` and a simple retry mechanism to access an API.

```ini
[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "gpt-3.5-turbo", "temperature": 0.3}
```

| Argument    | Type             | Default | Description                                                                                                          |
| ----------- | ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `api`       | `str`            |         | The name of a supported API. In v.0.1.0, only "OpenAI" is supported.                                                 |
| `config`    | `Dict[Any, Any]` | `{}`    | Further configuration passed on to the backend.                                                                      |
| `strict`    | `bool`           | `True`  | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`            | `3`     | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`            | `30`    | Timeout for API request in seconds.                                                                                  |

When `api` is set to `OpenAI`, the following settings can be defined in the `config` dictionary:

- `model`: one of the following list of supported models:
  - `"gpt-4"`
  - `"gpt-4-0314"`
  - `"gpt-4-32k"`
  - `"gpt-4-32k-0314"`
  - `"gpt-3.5-turbo"`
  - `"gpt-3.5-turbo-0301"`
  - `"text-davinci-003"`
  - `"text-davinci-002"`
  - `"text-curie-001"`
  - `"text-babbage-001"`
  - `"text-ada-001"`
  - `"davinci"`
  - `"curie"`
  - `"babbage"`
  - `"ada"`
- `url`: By default, this is `https://api.openai.com/v1/completions`. For models requiring the chat endpoint, use `https://api.openai.com/v1/chat/completions`.

#### spacy.MiniChain.v1

To use [MiniChain](https://github.com/srush/MiniChain) for the API retrieval part, make sure you have installed it first:

```shell
python -m pip install "minichain>=0.3,<0.4"
# Or install with spacy-llm directly
python -m pip install "spacy-llm[minichain]"
```

Note that MiniChain currently only supports Python 3.8, 3.9 and 3.10.

Example config blocks:

```ini
[components.llm.backend]
@llm_backends = "spacy.MiniChain.v1"
api = "OpenAI"

[components.llm.backend.query]
@llm_queries = "spacy.RunMiniChain.v1"
```

| Argument | Type                                                                              | Default | Description                                                                         |
| -------- | --------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `api`    | `str`                                                                             |         | The name of an API supported by MiniChain, e.g. "OpenAI".                           |
| `config` | `Dict[Any, Any]`                                                                  | `{}`    | Further configuration passed on to the backend.                                     |
| `query`  | `Optional[Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]]` | `None`  | Function that executes the prompts. If `None`, defaults to `spacy.RunMiniChain.v1`. |

The default `query` (`spacy.RunMiniChain.v1`) executes the prompts by running `model(text).run()` for each given textual prompt.

#### spacy.LangChain.v1

To use [LangChain](https://github.com/hwchase17/langchain) for the API retrieval part, make sure you have installed it first:

```shell
python -m pip install "langchain>=0.0.144,<0.1"
# Or install with spacy-llm directly
python -m pip install "spacy-llm[langchain]"
```

Note that LangChain currently only supports Python 3.9 and beyond.

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.LangChain.v1"
api = "OpenAI"
query = {"@llm_queries": "spacy.CallLangChain.v1"}
config = {"temperature": 0.3}
```

| Argument | Type                                                                           | Default | Description                                                                          |
| -------- | ------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------ |
| `api`    | `str`                                                                          |         | The name of an API supported by LangChain, e.g. "OpenAI".                            |
| `config` | `Dict[Any, Any]`                                                               | `{}`    | Further configuration passed on to the backend.                                      |
| `query`  | `Optional[Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]]` | `None`  | Function that executes the prompts. If `None`, defaults to `spacy.CallLangChain.v1`. |

The default `query` (`spacy.CallLangChain.v1`) executes the prompts by running `model(text)` for each given textual prompt.

#### spacy.Dolly_HF.v1

To use this backend, ideally you have a GPU enabled and have installed `transformers`, `torch` and CUDA in your virtual environment.
This allows you to have the setting `device=cuda:0` in your config, which ensures that the model is loaded entirely on the GPU (and fails otherwise).

You can do so with

```shell
python -m pip install "spacy-llm[transformers]" "transformers[sentencepiece]"
```

If you don't have access to a GPU, you can install `accelerate` and set`device_map=auto` instead, but be aware that this may result in some layers getting distributed to the CPU or even the hard drive,
which may ultimately result in extremely slow queries.

```shell
python -m pip install "accelerate>=0.16.0,<1.0"
```

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.Dolly_HF.v1"
model = "databricks/dolly-v2-3b"
```

| Argument      | Type             | Default | Description                                                                                      |
| ------------- | ---------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `model`       | `str`            |         | The name of a Dolly model that is supported.                                                     |
| `config_init` | `Dict[str, Any]` | `{}`    | Further configuration passed on to the construction of the model with `transformers.pipeline()`. |
| `config_run`  | `Dict[str, Any]` | `{}`    | Further configuration used during model inference.                                               |

Supported models (see the [Databricks models page](https://huggingface.co/databricks) on Hugging Face for details):

- `"databricks/dolly-v2-3b"`
- `"databricks/dolly-v2-7b"`
- `"databricks/dolly-v2-12b"`

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### spacy.StableLM_HF.v1

To use this backend, ideally you have a GPU enabled and have installed `transformers`, `torch` and CUDA in your virtual environment.

You can do so with

```shell
python -m pip install "spacy-llm[transformers]" "transformers[sentencepiece]"
```

If you don't have access to a GPU, you can install `accelerate` and set`device_map=auto` instead, but be aware that this may result in some layers getting distributed to the CPU or even the hard drive,
which may ultimately result in extremely slow queries.

```shell
python -m pip install "accelerate>=0.16.0,<1.0"
```

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.StableLM_HF.v1"
model = "stabilityai/stablelm-tuned-alpha-7b"
```

| Argument      | Type             | Default | Description                                                                                                                  |
| ------------- | ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `str`            |         | The name of a StableLM model that is supported.                                                                              |
| `config_init` | `Dict[str, Any]` | `{}`    | Further configuration passed on to the construction of the model with `transformers.AutoModelForCausalLM.from_pretrained()`. |
| `config_run`  | `Dict[str, Any]` | `{}`    | Further configuration used during model inference.                                                                           |

Supported models (see the [Stability AI StableLM GitHub repo](https://github.com/Stability-AI/StableLM/#stablelm-alpha) for details):

- `"stabilityai/stablelm-base-alpha-3b"`
- `"stabilityai/stablelm-base-alpha-7b"`
- `"stabilityai/stablelm-tuned-alpha-3b"`
- `"stabilityai/stablelm-tuned-alpha-7b"`

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### spacy.OpenLLaMaHF.v1

To use this backend, ideally you have a GPU enabled and have installed

- `transformers[sentencepiece]`
- `torch`
- CUDA
  in your virtual environment.

You can do so with

```shell
python -m pip install "spacy-llm[transformers]" "transformers[sentencepiece]"
```

If you don't have access to a GPU, you can install `accelerate` and set`device_map=auto` instead, but be aware that this may result in some layers getting distributed to the CPU or even the hard drive,
which may ultimately result in extremely slow queries.

```shell
python -m pip install "accelerate>=0.16.0,<1.0"
```

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.OpenLLaMaHF.v1"
model = "openlm-research/open_llama_3b_350bt_preview"
```

| Argument      | Type             | Default | Description                                                                                                                  |
| ------------- | ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `str`            |         | The name of a OpenLLaMa model that is supported.                                                                             |
| `config_init` | `Dict[str, Any]` | `{}`    | Further configuration passed on to the construction of the model with `transformers.AutoModelForCausalLM.from_pretrained()`. |
| `config_run`  | `Dict[str, Any]` | `{}`    | Further configuration used during model inference.                                                                           |

Supported models (see the [OpenLM Research OpenLLaMa GitHub repo](https://github.com/openlm-research/open_llama) for details):

- `"openlm-research/open_llama_3b_350bt_preview"`
- `"openlm-research/open_llama_3b_600bt_preview"`
- `"openlm-research/open_llama_7b_400bt_preview"`
- `"openlm-research/open_llama_7b_700bt_preview"`

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

### Cache

Interacting with LLMs, either through an external API or a local instance, is costly.
Since developing an NLP pipeline generally means a lot of exploration and prototyping,
`spacy-llm` implements a built-in cache to avoid reprocessing the same documents at each run
that keeps batches of documents stored on disk.

Example config block:

```ini
[components.llm.cache]
@llm_misc = "spacy.BatchCache.v1"
path = "path/to/cache"
batch_size = 64
max_batches_in_mem = 4
```

| Argument             | Type                         | Default | Description                                                                                                               |
| -------------------- | ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `path`               | `Optional[Union[str, Path]]` | `None`  | Cache directory. If `None`, no caching is performed, and this component will act as a NoOp.                               |
| `batch_size`         | `int`                        | 64      | Number of docs in one batch (file). Once a batch is full, it will be peristed to disk.                                    |
| `max_batches_in_mem` | `int`                        | 4       | Max. number of batches to hold in memory. Allows you to limit the effect on your memory if you're handling a lot of docs. |

When retrieving a document, the `BatchCache` will first figure out what batch the document belongs to. If the batch
isn't in memory it will try to load the batch from disk and then move it into memory. 

Note that since the cache is generated by a registered function, you can also provide your own registered function
returning your own cache implementation. If you wish to do so, ensure that your cache object adheres to the
`Protocol` defined in `spacy_llm.ty.Cache`.

### Various functions

#### spacy.FewShotReader.v1

This function is registered in spaCy's `misc` registry, and reads in examples from a `.yml`, `.yaml`, `.json` or `.jsonl` file.
It uses [`srsly`](https://github.com/explosion/srsly) to read in these files and parses them depending on the file extension.

```ini
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

| Argument | Type               | Description                                                                |
| -------- | ------------------ | -------------------------------------------------------------------------- |
| `path`   | `Union[str, Path]` | Path to an examples file with suffix `.yml`, `.yaml`, `.json` or `.jsonl`. |

#### spacy.FileReader.v1

This function is registered in spaCy's `misc` registry, and reads a file provided to the `path` to return a `str`
representation of its contents. This function is typically used to read
[Jinja](https://jinja.palletsprojects.com/en/3.1.x/) files containing the prompt template.

```ini
[components.llm.task.template]
@misc = "spacy.FileReader.v1"
path = "ner_template.jinja2"
```

| Argument | Type               | Description                  |
| -------- | ------------------ | ---------------------------- |
| `path`   | `Union[str, Path]` | Path to the file to be read. |

#### Normalizer functions

These functions provide simple normalizations for string comparisons, e.g. between a list of specified labels
and a label given in the raw text of the LLM response. They are registered in spaCy's `misc` registry
and have the signature `Callable[[str], str]`.

- `spacy.StripNormalizer.v1`: only apply `text.strip()`
- `spacy.LowercaseNormalizer.v1`: applies `text.strip().lower()` to compare strings in a case-insensitive way.

## 🚀 Ongoing work

In the near future, we will

- Add more example tasks
- Support a broader range of models
- Provide more example use-cases and tutorials
- Make the built-in tasks easier to customize via Jinja templates to define the instructions & examples

PRs are always welcome!

## 📝️ Reporting issues

If you have questions regarding the usage of `spacy-llm`, or want to give us feedback after giving it a spin, please use the
[discussion board](https://github.com/explosion/spaCy/discussions).
Bug reports can be filed on the [spaCy issue tracker](https://github.com/explosion/spaCy/issues). Thank you!
