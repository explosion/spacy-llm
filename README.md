# spacy-llm: Integrating LLMs into structured NLP pipelines

This package supports integration of Large Language Models (LLMs) into [spaCy](https://spacy.io/).
It adds an `llm` pipeline component to spaCy, allowing to prompt LLMs as part of your spaCy pipeline.
`llm` behaves like any other pipeline component and is (de-)serializable.

Each `llm` component is defined by two main settings:

- A [_Task_](#Tasks), defining the prompt to send to the LLM as well as the functionality to parse the resulting response
  back into structured fields on spaCy's [Doc](https://spacy.io/api/doc) objects.
- A [_Backend_](#Backends) defining the model to use and how to connect to it. Note that `spacy-llm` supports both access to external
  APIs (such as OpenAI) as well as access to self-hosted open-source LLMs (such as using Dolly through HuggingFace).

## ⏳ Install

`spacy-llm` will be installed automatically from spaCy v.3.5.3 onwards. For older spaCy v3 versions, you can run

```bash
pip install spacy-llm
```

in the same virtual environment where you already have `spacy` [installed](https://spacy.io/usage).

## 🐍 Usage

The task and the backend have to be supplied to the `llm` pipeline component using [spaCy's config
system](https://spacy.io/api/data-formats#config). This package provides various built-in
functionality, documented below in the [API](#📓-API) section.

### Example 1: run NER using an open-source model through HuggingFace

To run this example, ensure that you have a GPU enabled, and `transformers`, `torch` and CUDA installed.
For more background information, see the [DollyHF](#DollyHF) section.

Create a config file `config.cfg` containing at least the following
(or see the full example [here](usage_examples/dolly_ner_zeroshot.cfg)):

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION

[components.llm.backend]
@llm_backends = "spacy.DollyHF.v1"
model = "databricks/dolly-v2-12b"
```

Now run:

```python
from spacy import util

config = util.load_config("config.cfg")
nlp = util.load_model_from_config(config, auto_fill=True)
doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
print([(ent.text, ent.label_) for ent in doc.ents])
```

Note that HuggingFace will download the `"databricks/dolly-v2-3b"` model the first time you use it. You can
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache)
by setting the environmental variable `HF_HOME`.
Also, you can upgrade the model to be `"databricks/dolly-v2-12b"` for better performance.

### Example 2: run TextCat using a GPT-3 model from OpenAI

To run this example, ensure that you `openai` installed.
Create a new API key from openai.com or fetch an existing one, and ensure the keys are set as environmental variables.
For more background information, see the [OpenAI](#OpenAI) section.

Create a config file `config.cfg` containing at least the following
(or see the full example [here](usage_examples/openai_textcat_zeroshot.cfg)):

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = COMPLIMENT,INSULT

[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "text-davinci-003", "temperature": 0.3}
```

Now run:

```python
from spacy import util

config = util.load_config("config.cfg")
nlp = util.load_model_from_config(config, auto_fill=True)
doc = nlp("You look gorgeous!")
print(doc.cats)
```

### Example 3: creating the component directly in Python

The `llm` component behaves as any other spaCy component does, so adding it to an existing pipeline follows the same
pattern:

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe(
    "llm",
    config={
        "task": {
            "@llm_tasks": "spacy.NER.v1",
            "labels": "PERSON,ORGANISATION,LOCATION"
        },
        "backend": {
            "@llm_backends": "spacy.REST.v1",
            "api": "OpenAI",
            "config": {"model": "text-davinci-003"},
        },
    },
)
doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
print([(ent.text, ent.label_) for ent in doc.ents])
```

Note that for efficient usage of resources, typically you would use [`nlp.pipe(docs)`](https://spacy.io/api/language#pipe)
with a batch, instead of calling `nlp(doc)` with a single document.

## ⚠️ Warning: experimental package

This package is experimental and it is possible that changes made to the interface will be breaking in minor version updates.

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

## 📓 API

### Tasks

A _task_ defines an NLP problem or question, that will be sent to the LLM via a prompt. Further, the task defines
how to parse the LLM's responses back into structured information. All tasks are registered in spaCy's `llm_tasks` registry.

Practically speaking, a task should adhere to the `LLMTask` `Protocol` defined in [ty.py](https://github.com/explosion/spacy-llm/blob/main/spacy_llm/ty.py).
It needs to define a `generate_prompts` function and a `parse_responses` function.

#### <kbd>function</kbd> `task.generate_prompts`

Takes a collection of documents, and returns a collection of "prompts", which can be of type `Any`.
Often, prompts are of type `str` but this is not enforced to allow for maximum flexibility in the framework.

| Argument    | Type          | Description            |
| ----------- | ------------- | ---------------------- |
| `docs`      | Iterable[Doc] | The input documents.   |
| **RETURNS** | Iterable[Any] | The generated prompts. |

#### <kbd>function</kbd> `task.parse_responses`

Takes a collection of LLM responses and the original documents, parses the responses into structured information,
and sets the annotations on the documents. The `parse_responses` function is free to set the annotations in any way,
including `Doc` fields like `ents`, `spans` or `cats`, or using custom defined fields.

The `responses` are of type `Iterable[Any]`, though they will often be `str` objects. This depends on the
return type of the [backend](#backends).

| Argument    | Type          | Description              |
| ----------- | ------------- | ------------------------ |
| `docs`      | Iterable[Doc] | The input documents.     |
| `responses` | Iterable[Any] | The generated prompts.   |
| **RETURNS** | Iterable[Doc] | The annotated documents. |

#### spacy.NER.v1

The NER task is a default implementation, adhering to the `LLMTask` protocol. It supports both zero-shot and
few-shot prompting.

```
[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION
examples = null
```

| Argument                  | Type                                  | Default      | Description                                                                                                                                  |
| ------------------------- | ------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | str                                   |              | Comma-separated list of labels.                                                                                                              |
| `examples`                | Optional[Callable[[], Iterable[Any]]] | `None`       | Optional function that generates examples for few-shot learning.                                                                             |
| `normalizer`              | Optional[Callable[[str], str]]        | `None`       | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                            |
| `alignment_mode`          | str                                   | `"contract"` | Alignment mode in case the LLM returns entities that do not align with token boundaries. Options are `"strict"`, `"contract"` or `"expand"`. |
| `case_sensitive_matching` | bool                                  | `False`      | Whether to search without case sensitivity.                                                                                                  |
| `single_match`            | bool                                  | `False`      | Whether to match an entity in the LLM's response only once (the first hit) or multiple times.                                                |

The NER task implementation doesn't currently ask specific offsets from the LLM, but simply expects a list of strings that represent the enties in the document.
This means that a form of string matching is required. This can be configured by the following parameters:

- The `single_match` parameter is typically set to `False` to allow for multiple matches. For instance, the response from the LLM might only mention the entity "Paris" once, but you'd still
  want to mark it every time it occurs in the document.
- The case-sensitive matching is typically set to `False` to be robust against case variances in the LLM's output.
- The `alignment_mode` argument is used to match entities as returned by the LLM to the tokens from the original `Doc` - specifically it's used as argument
  in the call to [`doc.char_span()`](https://spacy.io/api/doc#char_span). The `"strict"` mode will only keep spans that strictly adhere to the given token boundaries.
  `"contract"` will only keep those tokens that are fully within the given range, e.g. reducing `"New Y"` to `"New"`.
  Finally, `"expand"` will expand the span to the next token boundaries, e.g. expanding `"New Y"` out to `"New York"`.

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` or `.jsonl`.

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

```
[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION

[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

#### spacy.TextCat.v1

The TextCat task is a default implementation, adhering to the `LLMTask` protocol. It supports both zero-shot and
few-shot prompting.

```
[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = COMPLIMENT,INSULT
examples = null
```

| Argument            | Type                                  | Default | Description                                                                                                              |
| ------------------- | ------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| `labels`            | str                                   |         | Comma-separated list of labels.                                                                                          |
| `examples`          | Optional[Callable[[], Iterable[Any]]] | `None`  | Optional function that generates examples for few-shot learning.                                                         |
| `normalizer`        | Optional[Callable[[str], str]]        | `None`  | Function that normalizes the labels as returned by the LLM. If `None`, falls back to `spacy.LowercaseNormalizer.v1`.     |
| `exclusive_classes` | bool                                  | `False` | If set to `True`, only one label per document should be valid. If set to `False`, one document can have multiple labels. |
| `verbose`           | bool                                  | `False` | If set to `True`, warnings will be generated when the LLM returns invalid responses.                                     |

To perform few-shot learning, you can write down a few examples in a separate file, and provide these to be injected into the prompt to the LLM.
The default reader `spacy.FewShotReader.v1` supports `.yml`, `.yaml`, `.json` or `.jsonl`.

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

```
[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = COMPLIMENT,INSULT

[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "textcat_examples.json"
```

#### spacy.NoOp.v1

This task is only useful for testing - it tells the LLM to do nothing, and does not set any fields on the `docs`.

```
[components.llm.task]
@llm_tasks = "spacy.NoOp.v1"
```

### Backends

A _backend_ defines which LLM model to query, and how to query it. It can be a simple function taking a collection
of prompts (consistent with the output type of `task.generate_prompts()`) and returning a collection of responses
(consistent with the expected input of `parse_responses`). Generally speaking, it's a function of type `Callable[[Iterable[Any]], Iterable[Any]]`,
but specific implementations can have other signatures, like `Callable[[Iterable[str]], Iterable[str]]`.

All built-in backends are registered in `llm_backends`. If no backend is specified, the repo currently connects to the [`OpenAI` API](#openai) by default,
using the built-in REST protocol, and accesses the `"text-davinci-003"` model.

#### OpenAI

When the backend uses OpenAI, you have to get an API key from openai.com, and ensure that the keys are set as
environmental variables. For instance, set a `.env` file in the root of your directory with the following information,
and make sure to exclude this file from git versioning:

```
OPENAI_ORG = "org-..."
OPENAI_API_KEY = "sk-..."
```

#### spacy.REST.v1

This default backend uses `requests` and a relatively simple retry mechanism to access an API.

```
[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "text-davinci-003", "temperature": 0.3}
```

| Argument    | Type           | Default | Description                                                                                                          |
| ----------- | -------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `api`       | str            |         | The name of a supported API. In v.0.1.0, only "OpenAI" is supported.                                                 |
| `config`    | Dict[Any, Any] | `{}`    | Further configuration passed on to the backend.                                                                      |
| `strict`    | bool           | `True`  | If `True`, raises an error if the LLM API returns a malformed response. Otherwise, return the error responses as is. |
| `max_tries` | int            | `3`     | Max. number of tries for API request.                                                                                |
| `timeout`   | int            | `30`    | Timeout for API request in seconds.                                                                                  |

When the `api` is set to `OpenAI`, the following settings can be defined in the `config` dictionary:

- `model`: one of the following list of supported models:
  - "text-davinci-003"
  - "text-davinci-002"
  - "text-curie-001"
  - "text-babbage-001"
  - "text-ada-001"
  - "davinci"
  - "curie"
  - "babbage"
  - "ada"
- `url`: By default, this is `https://api.openai.com/v1/completions`

#### spacy.MiniChain.v1

TODO

#### spacy.LangChain.v1

TODO

#### spacy.DollyHF.v1

TODO 

Supported models:
- `"databricks/dolly-v2-3b"`
- `"databricks/dolly-v2-7b"`
- `"databricks/dolly-v2-12b"`

See the [Databricks models page](https://huggingface.co/databricks) on HuggingFace for details.

#### OpenAI

TODO

```
OPENAI_ORG = "org-..."
OPENAI_API_KEY = "sk-..."
```