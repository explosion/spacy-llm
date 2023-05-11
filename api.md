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

```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v1"
labels = PERSON,ORGANISATION,LOCATION
examples = null
```

| Argument                  | Type                                  | Default      | Description                                                                                                                                  |
| ------------------------- | ------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`                  | str                                   |              | Comma-separated list of labels.                                                                                                              |
| `examples`                | Optional[Callable[[], Iterable[Any]]] | `None`       | Optional function that generates examples for few-shot learning.                                                                             |
| `normalizer`              | Optional[Callable[[str], str]]        | `None`       | Function that normalizes the labels as returned by the LLM. If `None`, defaults to `spacy.LowercaseNormalizer.v1`.                           |
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

```ini
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

```ini
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

```ini
[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = COMPLIMENT,INSULT
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "textcat_examples.json"
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
using the built-in REST protocol, and accesses the `"text-davinci-003"` model.

#### OpenAI

When the backend uses OpenAI, you have to get an API key from openai.com, and ensure that the keys are set as
environmental variables. For instance, set a `.env` file in the root of your directory with the following information,
and make sure to exclude this file from git versioning:

```bash
OPENAI_ORG = "org-..."
OPENAI_API_KEY = "sk-..."
```

#### spacy.REST.v1

This default backend uses `requests` and a relatively simple retry mechanism to access an API.

```ini
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
  - `"text-davinci-003"`
  - `"text-davinci-002"`
  - `"text-curie-001"`
  - `"text-babbage-001"`
  - `"text-ada-001"`
  - `"davinci"`
  - `"curie"`
  - `"babbage"`
  - `"ada"`
- `url`: By default, this is `https://api.openai.com/v1/completions`

#### spacy.MiniChain.v1

To use [MiniChain](https://github.com/srush/MiniChain) for the API retrieval part, make sure you have installed it first:

```
python -m pip install "minichain>=0.3,<0.4"
```

Note that MiniChain currently only supports Python 3.8, 3.9 and 3.10.

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.MiniChain.v1"
api = "OpenAI"
"query": {"@llm_queries": "spacy.RunMiniChain.v1"},
```

| Argument | Type                                                                            | Default | Description                                                                         |
| -------- | ------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `api`    | str                                                                             |         | The name of an API supported by MiniChain, e.g. "OpenAI".                           |
| `config` | Dict[Any, Any]                                                                  | `{}`    | Further configuration passed on to the backend.                                     |
| `query`  | Optional[Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]] | `None`  | Function that executes the prompts. If `None`, defaults to `spacy.RunMiniChain.v1`. |

The default `query` `spacy.RunMiniChain.v1` executes the prompts by running `model(text).run()` for each given textual prompt.

#### spacy.LangChain.v1

To use [LangChain](https://github.com/hwchase17/langchain) for the API retrieval part, make sure you have installed it first:

```
python -m pip install "langchain>=0.0.144,<0.1"
```

Note that LangChain currently only supports Python 3.9 and beyond.

Example config block:

```
[components.llm.backend]
@llm_backends = "spacy.LangChain.v1"
api = "OpenAI"
query = {"@llm_queries": "spacy.CallLangChain.v1"},
config = {"temperature": 0.3},
```

| Argument | Type                                                                         | Default | Description                                                                          |
| -------- | ---------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `api`    | str                                                                          |         | The name of an API supported by LangChain, e.g. "OpenAI".                            |
| `config` | Dict[Any, Any]                                                               | `{}`    | Further configuration passed on to the backend.                                      |
| `query`  | Optional[Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]] | `None`  | Function that executes the prompts. If `None`, defaults to `spacy.CallLangChain.v1`. |

The default `query` `spacy.CallLangChain.v1` executes the prompts by running `model(text)` for each given textual prompt.

#### spacy.DollyHF.v1

To use this backend, ideally you have a GPU enabled and have installed `transformers`, `torch` and CUDA in your virtual environment.
This allows you to have the setting `device=cuda:0` in your config, which ensures that the model is loaded entirely on the GPU (and fails otherwise).

```
python -m pip install "cupy-cuda11x"
python -m pip install "torch>=1.13.1,<2.0"
python -m pip install "transformers>=4.28.1,<5.0"
```

If you don't have access to a GPU, you can install `accelerate` and set`device_map=auto` instead, but be aware that this may result in some layers getting distributed to the CPU or even the hard drive,
which may ultimately result in extremely slow queries.

```
python -m pip install "accelerate>=0.16.0,<1.0"
```

Example config block:

```ini
[components.llm.backend]
@llm_backends = "spacy.DollyHF.v1"
model = "databricks/dolly-v2-3b"
```

| Argument | Type           | Default | Description                                                                                      |
| -------- | -------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `model`  | str            |         | The name of a Dolly model that is supported.                                                     |
| `config` | Dict[Any, Any] | `{}`    | Further configuration passed on to the construction of the model with `transformers.pipeline()`. |

Supported models (see the [Databricks models page](https://huggingface.co/databricks) on HuggingFace for details):

- `"databricks/dolly-v2-3b"`
- `"databricks/dolly-v2-7b"`
- `"databricks/dolly-v2-12b"`

Note that HuggingFace will download this model the first time you use it - you can 
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache) 
by setting the environmental variable `HF_HOME`. 

### Various functions

#### spacy.FewShotReader.v1

This function is registered in spaCy's `misc` registry, and reads in examples from a `.yml`, `.yaml`, `.json` or `.jsonl` file.
It uses [`srsly`](https://github.com/explosion/srsly) to read in these files and parses them depending on the file extension.

```ini
[components.llm.task.examples]
@misc = "spacy.FewShotReader.v1"
path = "ner_examples.yml"
```

| Argument | Type             | Description                                                                |
| -------- | ---------------- | -------------------------------------------------------------------------- |
| `path`   | Union[str, Path] | Path to an examples file with suffix `.yml`, `.yaml`, `.json` or `.jsonl`. |

#### Normalizer functions

These functions provide simple normalizations for string comparisons, e.g. between a list of specified labels 
and a label given in the raw text of the LLM response. They are registered in spaCy's `misc` registry 
and have the signature `Callable[[str], str]`.
* `spacy.StripNormalizer.v1`: only apply `text.strip()`
* `spacy.LowercaseNormalizer.v1`: applies `text.strip().lower()` to compare strings in a case-insensitive way.