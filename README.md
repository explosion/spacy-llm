<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-llm: Integrating LLMs into structured NLP pipelines

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/explosion/spacy-llm/test.yml?branch=main)](https://github.com/explosion/spacy-llm/actions/workflows/test.yml)
[![pypi Version](https://img.shields.io/pypi/v/spacy-llm.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spacy-llm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

This package integrates Large Language Models (LLMs) into [spaCy](https://spacy.io), featuring a modular system for **fast prototyping** and **prompting**, and turning unstructured responses into **robust outputs** for various NLP tasks, **no training data** required.

- Serializable `llm` **component** to integrate prompts into your pipeline
- **Modular functions** to define the [**task**](#tasks) (prompting and parsing) and [**model**](#models)
- Support for **hosted APIs** and self-hosted **open-source models**
- Integration with [`LangChain`](https://github.com/hwchase17/langchain)
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

The task and the model have to be supplied to the `llm` pipeline component using [spaCy's config
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

[components.llm.model]
@llm_models = "spacy.GPT-3-5.v1"
config = {"temperature": 0.3}
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

[components.llm.model]
@llm_models = "spacy.Dolly.v1"
# For better performance, use dolly-v2-12b instead
name = "dolly-v2-3b"
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
        "model": {
            "@llm_models": "spacy.gpt-3.5.v1",
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

Model response for doc: You look gorgeous!
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
| `model`          | `Callable[[Iterable[Any]], Iterable[Any]]]` | Callable querying a specific LLM API. See [docs](#models).                          |
| `cache`          | `Cache`                                     | Cache to use for caching prompts and responses per doc (batch). See [docs](#cache). |
| `save_io`        | `bool`                                      | Whether to save prompts/responses within `Doc.user_data["llm_io"]`.                 |
| `validate_types` | `bool`                                      | Whether to check if signatures of configured model and task are consistent.         |

An `llm` component is defined by two main settings:

- A [**task**](#tasks), defining the prompt to send to the LLM as well as the functionality to parse the resulting response
  back into structured fields on spaCy's [Doc](https://spacy.io/api/doc) objects.
- A [**model**](#models) defining the model and how to connect to it. Note that `spacy-llm` supports both access to external
  APIs (such as OpenAI) as well as access to self-hosted open-source LLMs (such as using Dolly through Hugging Face).

Moreover, `spacy-llm` exposes a customizable [**caching**](#cache) functionality to avoid running
the same document through an LLM service (be it local or through a REST API) more than once.

Finally, you can choose to save a stringified version of LLM prompts/responses
within the `Doc.user_data["llm_io"]` attribute by setting `save_io` to `True`.
`Doc.user_data["llm_io"]` is a dictionary containing one entry for every LLM component
within the spaCy pipeline. Each entry is itself a dictionary, with two keys:
`prompt` and `response`.

A note on `validate_types`: by default, `spacy-llm` checks whether the signatures of the `model` and `task` callables
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

#### Providing examples for few-shot prompts

All built-in tasks support few-shot prompts, i. e. including examples in a prompt. Examples can be supplied in two ways:
(1) as a separate file containing only examples or (2) by initializing `llm` with a `get_examples()` callback (like any
other spaCy pipeline component).

##### (1) Few-shot example file

A file containing examples for few-shot prompting can be configured like this:

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = PERSON,ORGANISATION,LOCATION
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

The supplied file has to conform to the format expected by the required task (see the task documentation further down).

##### (2) Initializing the `llm` component with a `get_examples()` callback

Alternatively, you can initialize your `nlp` pipeline by providing a `get_examples` callback for
[`nlp.initialize`](https://spacy.io/api/language#initialize) and setting `n_prompt_examples` to a positive number to
automatically fetch a few examples for few-shot learning. Set `n_prompt_examples` to `-1` to use all examples as
part of the few-shot learning prompt.

```ini
[initialize.components.llm]
n_prompt_examples = 3
```

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
return type of the [model](#models).

| Argument    | Type            | Description              |
| ----------- | --------------- | ------------------------ |
| `docs`      | `Iterable[Doc]` | The input documents.     |
| `responses` | `Iterable[Any]` | The generated prompts.   |
| **RETURNS** | `Iterable[Doc]` | The annotated documents. |


#### spacy.Summarization.v1

The `spacy.Summarization.v1` task supports both zero-shot and few-shot prompting.

```ini
[components.llm.task]
@llm_tasks = "spacy.Summarization.v1"
examples = null
max_n_words = null
```

| Argument      | Type                                    | Default                                                                | Description                                                                                                                              |
|---------------|-----------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `template`    | `str`                                   | [summarization.jinja](./spacy_llm/tasks/templates/summarization.jinja) | Custom prompt template to send to LLM backend. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `examples`    | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                                 | Optional function that generates examples for few-shot learning.                                                                         |
| `max_n_words` | `Optional[int]`                         | `None`                                                                 | Maximum number of words to be used in summary. Note that this should not expected to work exactly.                                       |
| `field`       | `str`                                   | `summary`                                                              | Name of extension attribute to store summary in (i. e. the summary will be available in `doc._.{field}`).                                |

The summarization task prompts the model for a concise summary of the provided text. It optionally allows to limit the 
response to a certain number of tokens - note that this requirement will be included in the prompt, but the task doesn't
perform a hard cut-off. It's hence possible that your summary exceeds `max_n_words`.

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```yaml
- text: >
    The United Nations, referred to informally as the UN, is an intergovernmental organization whose stated purposes are 
    to maintain international peace and security, develop friendly relations among nations, achieve international 
    cooperation, and serve as a centre for harmonizing the actions of nations. It is the world's largest international 
    organization. The UN is headquartered on international territory in New York City, and the organization has other 
    offices in Geneva, Nairobi, Vienna, and The Hague, where the International Court of Justice is headquartered.\n\n
    The UN was established after World War II with the aim of preventing future world wars, and succeeded the League of 
    Nations, which was characterized as ineffective. 
  summary: "The UN is an international organization that promotes global peace, cooperation, and harmony. Established after WWII, its purpose is to prevent future world wars."
```

```ini
[components.llm.task]
@llm_tasks = "spacy.summarization.v1"
max_n_words = 20
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "summarization_examples.yml"
```

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
| `template`                | `str`                                   | [ner.v2.jinja](./spacy_llm/tasks/templates/ner.v2.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.                |
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

You can also write definitions for each label and provide them via the `label_definitions` argument. This lets you tell
the LLM exactly what you're looking for rather than relying on the LLM to interpret its task given just the label name.
Label descriptions are freeform so you can write whatever you want here, but through some experiments a brief
description along with some examples and counter examples seems to work quite well.

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = PERSON,SPORTS_TEAM
[components.llm.task.label_definitions]
PERSON = "Extract any named individual in the text."
SPORTS_TEAM = "Extract the names of any professional sports team. e.g. Golden State Warriors, LA Lakers, Man City, Real Madrid"
```

> Label descriptions can also be used with explicit examples to give as much info to the LLM model as possible.

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
| `template`                | `str`                                   | [`spancat.v2.jinja`](./spacy_llm/tasks/templates/spancat.v2.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.                |
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

#### spacy.TextCat.v3

Version 3 (the most recent) of the built-in TextCat task supports both zero-shot and few-shot prompting. It allows
setting definitions of labels. Those definitions are included in the prompt.

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v3"
labels = ["COMPLIMENT", "INSULT"]
label_definitions = {
    "COMPLIMENT": "a polite expression of praise or admiration.",
    "INSULT": "a disrespectful or scornfully abusive remark or act."
}
examples = null
```

| Argument            | Type                                    | Default                                                      | Description                                                                                                                                      |
| ------------------- | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `labels`            | `Union[List[str], str]`                 |                                                              | List of labels or str of comma-separated list of labels.                                                                                         |
| `label_definitions` | `Optional[Dict[str, str]]`              | `None`                                                       | Dictionary of label definitions. Included in the prompt, if set.                                                                                 |
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
@llm_tasks = "spacy.TextCat.v3"
labels = ["COMPLIMENT", "INSULT"]
label_definitions = {
    "COMPLIMENT": "a polite expression of praise or admiration.",
    "INSULT": "a disrespectful or scornfully abusive remark or act."
}
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "textcat_examples.json"
```

#### spacy.TextCat.v2

Version 2 of the built-in TextCat task supports both zero-shot and few-shot prompting and includes an improved prompt
template.

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = ["COMPLIMENT", "INSULT"]
examples = null
```

| Argument            | Type                                    | Default                                                      | Description                                                                                                                                      |
| ------------------- | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `labels`            | `Union[List[str], str]`                 |                                                              | List of labels or str of comma-separated list of labels.                                                                                         |
| `template`          | `str`                                   | [`textcat.jinja`](./spacy_llm/tasks/templates/textcat.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory.           |
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

Version 1 of the built-in TextCat task supports both zero-shot and few-shot prompting.

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

| Argument            | Type                                    | Default                                              | Description                                                                                                                            |
| ------------------- | --------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`            | `Union[List[str], str]`                 |                                                      | List of labels or str of comma-separated list of labels.                                                                               |
| `template`          | `str`                                   | [`rel.jinja`](./spacy_llm/tasks/templates/rel.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `label_description` | `Optional[Dict[str, str]]`              | `None`                                               | Dictionary providing a description for each relation label.                                                                            |
| `examples`          | `Optional[Callable[[], Iterable[Any]]]` | `None`                                               | Optional function that generates examples for few-shot learning.                                                                       |
| `normalizer`        | `Optional[Callable[[str], str]]`        | `None`                                               | Function that normalizes the labels as returned by the LLM. If `None`, falls back to `spacy.LowercaseNormalizer.v1`.                   |
| `verbose`           | `bool`                                  | `False`                                              | If set to `True`, warnings will be generated when the LLM returns invalid responses.                                                   |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```jsonl
{"text": "Laura bought a house in Boston with her husband Mark.", "ents": [{"start_char": 0, "end_char": 5, "label": "PERSON"}, {"start_char": 24, "end_char": 30, "label": "GPE"}, {"start_char": 48, "end_char": 52, "label": "PERSON"}], "relations": [{"dep": 0, "dest": 1, "relation": "LivesIn"}, {"dep": 2, "dest": 1, "relation": "LivesIn"}]}
{"text": "Michael travelled through South America by bike.", "ents": [{"start_char": 0, "end_char": 7, "label": "PERSON"}, {"start_char": 26, "end_char": 39, "label": "LOC"}], "relations": [{"dep": 0, "dest": 1, "relation": "Visits"}]}
```

```ini
[components.llm.task]
@llm_tasks = "spacy.REL.v1"
labels = ["LivesIn", "Visits"]
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "rel_examples.jsonl"
```

Note: the REL task relies on pre-extracted entities to make its prediction.
Hence, you'll need to add a component that populates `doc.ents` with recognized
spans to your spaCy pipeline and put it _before_ the REL component.

#### spacy.Lemma.v1

The `Lemma.v1` task lemmatizes the provided text and updates the `lemma_` attribute in the doc's tokens accordingly.

```ini
[components.llm.task]
@llm_tasks = "spacy.Lemma.v1"
examples = null
```

| Argument   | Type                                    | Default                                                | Description                                                                                                                            |
| ---------- | --------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `template` | `str`                                   | [lemma.jinja](./spacy_llm/tasks/templates/lemma.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `examples` | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                 | Optional function that generates examples for few-shot learning.                                                                       |

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

#### spacy.Sentiment.v1

Performs sentiment analysis on provided texts. Scores between 0 and 1 are stored in `Doc._.sentiment` - the higher, the
more positive. Note in cases of parsing issues (e. g. in case of unexpected LLM responses) the value might be `None`.

```ini
[components.llm.task]
@llm_tasks = "spacy.Sentiment.v1"
examples = null
```

| Argument   | Type                                    | Default                                                        | Description                                                                                                                            |
| ---------- | --------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `template` | `str`                                   | [sentiment.jinja](./spacy_llm/tasks/templates/sentiment.jinja) | Custom prompt template to send to LLM model. Default templates for each task are located in the `spacy_llm/tasks/templates` directory. |
| `examples` | `Optional[Callable[[], Iterable[Any]]]` | `None`                                                         | Optional function that generates examples for few-shot learning.                                                                       |
| `field`    | `str`                                   | `sentiment`                                                    | Name of extension attribute to store summary in (i. e. the summary will be available in `doc._.{field}`).                              |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` and `.jsonl`.

```yaml
- text: "This is horrifying."
  score: 0
- text: "This is underwhelming."
  score: 0.25
- text: "This is ok."
  score: 0.5
- text: "I'm looking forward to this!"
  score: 1.0
```

```ini
[components.llm.task]
@llm_tasks = "spacy.Sentiment.v1"
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "sentiment_examples.yml"
```

#### spacy.NoOp.v1

This task is only useful for testing - it tells the LLM to do nothing, and does not set any fields on the `docs`.

```ini
[components.llm.task]
@llm_tasks = "spacy.NoOp.v1"
```

### Models

A _model_ defines which LLM model to query, and how to query it. It can be a simple function taking a collection
of prompts (consistent with the output type of `task.generate_prompts()`) and returning a collection of responses
(consistent with the expected input of `parse_responses`). Generally speaking, it's a function of type `Callable[[Iterable[Any]], Iterable[Any]]`,
but specific implementations can have other signatures, like `Callable[[Iterable[str]], Iterable[str]]`.

All built-in models are registered in `llm_models`. If no model is specified, the repo currently connects to the `OpenAI` API by default
using REST, and accesses the `"gpt-3.5-turbo"` model.

Currently three different approaches to use LLMs are supported:

1. `spacy-llm`s native REST backend. This is the default for all hosted models (e. g. OpenAI, Cohere, Anthropic, ...).
2. A HuggingFace integration that allows to run a limited set of HF models locally.
3. A LangChain integration that allows to run any model supported by LangChain (hosted or locally).

Approaches 1. and 2 are the default for hosted model and local models, respectively. Alternatively you can use LangChain
to access hosted or local models by specifying one of the models registered with the `langchain.` prefix.

> :question: Why LangChain if there are also are a native REST and a HuggingFace backend? When should I use what?
>
> Third-party libraries like `langchain` focus on prompt management, integration of many different LLM
> APIs, and other related features such as conversational memory or agents. `spacy-llm` on the other hand emphasizes
> features we consider useful in the context of NLP pipelines utilizing LLMs to process documents (mostly) independent
> from each other. It makes sense that the feature sets of such third-party libraries and `spacy-llm` aren't identical -
> and users might want to take advantage of features not available in `spacy-llm`.
>
> The advantage of implementing our own REST and HuggingFace integrations is that we can ensure a larger degree of stability and robustness, as
> we can guarantee backwards-compatibility and more smoothly integrated error handling.
>
> If however there are features or APIs not natively covered by `spacy-llm`, it's trivial to utilize LangChain to cover
> this - and easy to customize the prompting mechanism, if so required.

Note that when using hosted services, you have to ensure that the proper API keys are set as environment variables as
described by the corresponding provider's documentation.

E. g. when using OpenAI, you have to get an API key from openai.com, and ensure that the keys are set as
environmental variables:

```shell
export OPENAI_API_KEY="sk-..."
export OPENAI_API_ORG="org-..."
```

For Cohere it's

```shell
export CO_API_KEY="..."
```

and for Anthropic

```shell
export ANTHROPIC_API_KEY="..."
```

#### spacy.GPT-4.v1

OpenAI's `gpt-4` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.GPT-4.v1"
name = "gpt-4"
config = {"temperature": 0.3}
```

| Argument    | Type                                                            | Default   | Description                                                                                                          |
| ----------- | --------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"]` | `"gpt-4"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                                | `{}`      | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                                          | `True`    | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                                           | `3`       | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                                           | `30`      | Timeout for API request in seconds.                                                                                  |

#### spacy.GPT-3-5.v1

OpenAI's `gpt-3-5` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.GPT-3-5.v1"
name = "gpt-3.5-turbo"
config = {"temperature": 0.3}
```

| Argument    | Type                                                                                            | Default           | Description                                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-0613-16k"]` | `"gpt-3.5-turbo"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                                                                | `{}`              | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                                                                          | `True`            | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                                                                           | `3`               | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                                                                           | `30`              | Timeout for API request in seconds.                                                                                  |

#### spacy.Text-Davinci.v1

OpenAI's `text-davinci` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Text-Davinci.v1"
name = "text-davinci-003"
config = {"temperature": 0.3}
```

| Argument    | Type                                              | Default              | Description                                                                                                          |
| ----------- | ------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["text-davinci-002", "text-davinci-003"]` | `"text-davinci-003"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                  | `{}`                 | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                            | `True`               | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                             | `3`                  | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                             | `30`                 | Timeout for API request in seconds.                                                                                  |

#### spacy.Code-Davinci.v1

OpenAI's `code-davinci` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Code-Davinci.v1"
name = "code-davinci-002"
config = {"temperature": 0.3}
```

| Argument    | Type                          | Default              | Description                                                                                                          |
| ----------- | ----------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["code-davinci-002"]` | `"code-davinci-002"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`              | `{}`                 | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                        | `True`               | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                         | `3`                  | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                         | `30`                 | Timeout for API request in seconds.                                                                                  |

#### spacy.Text-Curie.v1

OpenAI's `text-curie` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Text-Curie.v1"
name = "text-curie-001"
config = {"temperature": 0.3}
```

| Argument    | Type                        | Default            | Description                                                                                                          |
| ----------- | --------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["text-curie-001"]` | `"text-curie-001"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`            | `{}`               | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                      | `True`             | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                       | `3`                | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                       | `30`               | Timeout for API request in seconds.                                                                                  |

#### spacy.Text-Babbage.v1

OpenAI's `text-babbage` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Text-Babbage.v1"
name = "text-babbage-001"
config = {"temperature": 0.3}
```

| Argument    | Type                          | Default              | Description                                                                                                          |
| ----------- | ----------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["text-babbage-001"]` | `"text-babbage-001"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`              | `{}`                 | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                        | `True`               | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                         | `3`                  | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                         | `30`                 | Timeout for API request in seconds.                                                                                  |

#### spacy.Text-Ada.v1

OpenAI's `text-ada` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Text-Ada.v1"
name = "text-ada-001"
config = {"temperature": 0.3}
```

| Argument    | Type                      | Default          | Description                                                                                                          |
| ----------- | ------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["text-ada-001"]` | `"text-ada-001"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`          | `{}`             | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                    | `True`           | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                     | `3`              | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                     | `30`             | Timeout for API request in seconds.                                                                                  |

#### spacy.Davinci.v1

OpenAI's `davinci` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Davinci.v1 "
name = "davinci"
config = {"temperature": 0.3}
```

| Argument    | Type                 | Default     | Description                                                                                                          |
| ----------- | -------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["davinci"]` | `"davinci"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`     | `{}`        | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`               | `True`      | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                | `3`         | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                | `30`        | Timeout for API request in seconds.                                                                                  |

#### spacy.Curie.v1

OpenAI's `curie` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Curie.v1 "
name = "curie"
config = {"temperature": 0.3}
```

| Argument    | Type               | Default   | Description                                                                                                          |
| ----------- | ------------------ | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["curie"]` | `"curie"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`   | `{}`      | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`             | `True`    | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`              | `3`       | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`              | `30`      | Timeout for API request in seconds.                                                                                  |

#### spacy.Babbage.v1

OpenAI's `babbage` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Babbage.v1 "
name = "babbage"
config = {"temperature": 0.3}
```

| Argument    | Type                 | Default     | Description                                                                                                          |
| ----------- | -------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["babbage"]` | `"babbage"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`     | `{}`        | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`               | `True`      | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                | `3`         | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                | `30`        | Timeout for API request in seconds.                                                                                  |

#### spacy.Ada.v1

OpenAI's `ada` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Ada.v1 "
name = "ada"
config = {"temperature": 0.3}
```

| Argument    | Type             | Default | Description                                                                                                          |
| ----------- | ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["ada"]` | `"ada"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]` | `{}`    | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`           | `True`  | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`            | `3`     | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`            | `30`    | Timeout for API request in seconds.                                                                                  |

#### spacy.Command.v1

Cohere's `command` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Command.v1 "
name = "command"
config = {"temperature": 0.3}
```

| Argument    | Type                                                                              | Default     | Description                                                                                                          |
| ----------- | --------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["command", "command-light", "command-light-nightly", "command-nightly"]` | `"command"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                                                  | `{}`        | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                                                            | `True`      | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                                                             | `3`         | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                                                             | `30`        | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-1.v1

Anthropic's `claude-1` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-1.v1 "
name = "claude-1"
config = {"temperature": 0.3}
```

| Argument    | Type                                   | Default      | Description                                                                                                          |
| ----------- | -------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-1", "claude-1-100k"]` | `"claude-1"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                       | `{}`         | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                 | `True`       | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                  | `3`          | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                  | `30`         | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-instant-1.v1

Anthropic's `claude-instant-1` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-instant-1.v1 "
name = "claude-instant-1"
config = {"temperature": 0.3}
```

| Argument    | Type                                                   | Default              | Description                                                                                                          |
| ----------- | ------------------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-instant-1", "claude-instant-1-100k"]` | `"claude-instant-1"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                       | `{}`                 | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                                 | `True`               | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                                  | `3`                  | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                                  | `30`                 | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-instant-1-1.v1

Anthropic's `claude-instant-1.1` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-instant-1-1.v1 "
name = "claude-instant-1.1"
config = {"temperature": 0.3}
```

| Argument    | Type                                                       | Default                | Description                                                                                                          |
| ----------- | ---------------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-instant-1.1", "claude-instant-1.1-100k"]` | `"claude-instant-1.1"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                                           | `{}`                   | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                                     | `True`                 | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                                      | `3`                    | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                                      | `30`                   | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-1-0.v1

Anthropic's `claude-1.0` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-1-0.v1 "
name = "claude-1.0"
config = {"temperature": 0.3}
```

| Argument    | Type                    | Default        | Description                                                                                                          |
| ----------- | ----------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-1.0"]` | `"claude-1.0"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`        | `{}`           | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                  | `True`         | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                   | `3`            | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                   | `30`           | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-1-2.v1

Anthropic's `claude-1.2` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-1-2.v1 "
name = "claude-1.2"
config = {"temperature": 0.3}
```

| Argument    | Type                    | Default        | Description                                                                                                          |
| ----------- | ----------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-1.2"]` | `"claude-1.2"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`        | `{}`           | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                  | `True`         | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                   | `3`            | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                   | `30`           | Timeout for API request in seconds.                                                                                  |

#### spacy.Claude-1-3.v1

Anthropic's `claude-1.3` model family.

Example config:

```ini
[components.llm.model]
@llm_models = "spacy.Claude-1-3.v1 "
name = "claude-1.3"
config = {"temperature": 0.3}
```

| Argument    | Type                                       | Default        | Description                                                                                                          |
| ----------- | ------------------------------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`      | `Literal["claude-1.3", "claude-1.3-100k"]` | `"claude-1.3"` | Model name, i. e. any supported variant for this particular model.                                                   |
| `config`    | `Dict[Any, Any]`                           | `{}`           | Further configuration passed on to the model.                                                                        |
| `strict`    | `bool`                                     | `True`         | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | `int`                                      | `3`            | Max. number of tries for API request.                                                                                |
| `timeout`   | `int`                                      | `30`           | Timeout for API request in seconds.                                                                                  |

#### spacy.Dolly.v1

To use this model, ideally you have a GPU enabled and have installed `transformers`, `torch` and CUDA in your virtual environment.
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
[components.llm.model]
@llm_models = "spacy.Dolly.v1"
name = "dolly-v2-3b"
```

| Argument      | Type                                                    | Default | Description                                                                                      |
| ------------- | ------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `name`        | `Literal["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]` |         | The name of a Dolly model that is supported (e. g. "dolly-v2-3b" or "dolly-v2-12b").             |
| `config_init` | `Dict[str, Any]`                                        | `{}`    | Further configuration passed on to the construction of the model with `transformers.pipeline()`. |
| `config_run`  | `Dict[str, Any]`                                        | `{}`    | Further configuration used during model inference.                                               |

Supported models (see the [Databricks models page](https://huggingface.co/databricks) on Hugging Face for details):

- `"databricks/dolly-v2-3b"`
- `"databricks/dolly-v2-7b"`
- `"databricks/dolly-v2-12b"`

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### spacy.Falcon.v1

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
[components.llm.model]
@llm_models = "spacy.Falcon.v1"
name = "falcon-7b"
```

| Argument      | Type                                                                                | Default         | Description                                                                                      |
| ------------- | ----------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------ |
| `name`        | `Literal["falcon-rw-1b", "falcon-7b", "falcon-7b-instruct", "falcon-40b-instruct"]` | `"7b-instruct"` | The name of a Falcon model variant that is supported.                                            |
| `config_init` | `Dict[str, Any]`                                                                    | `{}`            | Further configuration passed on to the construction of the model with `transformers.pipeline()`. |
| `config_run`  | `Dict[str, Any]`                                                                    | `{}`            | Further configuration used during model inference.                                               |

Note that Hugging Face will download this model the first time you use it - you can
[define the cache directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### spacy.StableLM.v1

To use this model, ideally you have a GPU enabled and have installed `transformers`, `torch` and CUDA in your virtual environment.

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
[components.llm.model]
@llm_models = "spacy.StableLM.v1"
name = "stablelm-tuned-alpha-7b"
```

| Argument      | Type                                                                                                                | Default | Description                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `Literal["stablelm-base-alpha-3b", "stablelm-base-alpha-7b", "stablelm-tuned-alpha-3b", "stablelm-tuned-alpha-7b"]` |         | The name of a StableLM model that is supported (e. g. "stablelm-tuned-alpha-7b").                                            |
| `config_init` | `Dict[str, Any]`                                                                                                    | `{}`    | Further configuration passed on to the construction of the model with `transformers.AutoModelForCausalLM.from_pretrained()`. |
| `config_run`  | `Dict[str, Any]`                                                                                                    | `{}`    | Further configuration used during model inference.                                                                           |

See the [Stability AI StableLM GitHub repo](https://github.com/Stability-AI/StableLM/#stablelm-alpha) for details.

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### spacy.OpenLLaMA.v1

To use this model, ideally you have a GPU enabled and have installed

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
[components.llm.model]
@llm_models = "spacy.OpenLLaMA.v1"
name = "open_llama_3b"
```

| Argument      | Type                                                                                            | Default | Description                                                                                                                  |
| ------------- |-------------------------------------------------------------------------------------------------| ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `Literal["open_llama_3b", "open_llama_7b", "open_llama_7b_v2", "open_llama_13b"]` |         | The name of a OpenLLaMA model that is supported.                                       |
| `config_init` | `Dict[str, Any]`                                                                                | `{}`    | Further configuration passed on to the construction of the model with `transformers.AutoModelForCausalLM.from_pretrained()`. |
| `config_run`  | `Dict[str, Any]`                                                                                | `{}`    | Further configuration used during model inference.                                                                           |

See the [OpenLM Research OpenLLaMA GitHub repo](https://github.com/openlm-research/open_llama) for details.

Note that Hugging Face will download this model the first time you use it - you can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.

#### LangChain models

To use [LangChain](https://github.com/hwchase17/langchain) for the API retrieval part, make sure you have installed it first:

```shell
python -m pip install "langchain==0.0.191"
# Or install with spacy-llm directly
python -m pip install "spacy-llm[extras]"
```

Note that LangChain currently only supports Python 3.9 and beyond.

LangChain models in `spacy-llm` work slightly differently. `langchain`'s models are parsed automatically, each
LLM class in `langchain` has one entry in `spacy-llm`'s registry. As `langchain`'s design has one class per API and
not per model, this results in registry entries like `langchain.OpenAI.v1` - i. e. there is one registry entry per API
and not per model (family), as for the REST- and HuggingFace-based entries.

The name of the model to be used has to be passed in via the `name` attribute.

Example config block:

```ini
[components.llm.model]
@llm_models = "langchain.OpenAI.v1"
name = "gpt-3.5-turbo"
query = {"@llm_queries": "spacy.CallLangChain.v1"}
config = {"temperature": 0.3}
```

| Argument | Type                                                                           | Default | Description                                                                          |
| -------- | ------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------ |
| `name`   | `str`                                                                          |         | The name of a mdodel supported by LangChain for this API.                            |
| `config` | `Dict[Any, Any]`                                                               | `{}`    | Configuration passed on to the LangChain model.                                      |
| `query`  | `Optional[Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]]` | `None`  | Function that executes the prompts. If `None`, defaults to `spacy.CallLangChain.v1`. |

The default `query` (`spacy.CallLangChain.v1`) executes the prompts by running `model(text)` for each given textual prompt.

### Cache

Interacting with LLMs, either through an external API or a local instance, is costly.
Since developing an NLP pipeline generally means a lot of exploration and prototyping,
`spacy-llm` implements a built-in cache to avoid reprocessing the same documents at each run
that keeps batches of documents stored on disk. 

The cache implementation also ensures that documents in one cache directory were all produced using the same prompt 
template. This is only possible however if the specified task implements 
```python
@property
def prompt_template() -> str:
    ...
``` 
which returns the raw prompt template as string. If `prompt_template()` isn't implemented, the cache will emit a warning
and not check for prompt template consistency.

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
| `batch_size`         | `int`                        | 64      | Number of docs in one batch (file). Once a batch is full, it will be persisted to disk.                                   |
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

## Migration guides

Please refer to our [migration guide](migration_guide.md).
