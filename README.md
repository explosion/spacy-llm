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

PRs are always welcome!

## 📝️ Reporting issues

If you have questions regarding the usage of `spacy-llm`, or want to give us feedback after giving it a spin, please use the
[discussion board](https://github.com/explosion/spaCy/discussions).
Bug reports can be filed on the [spaCy issue tracker](https://github.com/explosion/spaCy/issues). Thank you!

## 📓 API

### Tasks

TODO

#### spacy.NER.v1

TODO

#### spacy.TextCat.v1

TODO

### Backends

TODO

#### spacy.Rest.v1

TODO

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