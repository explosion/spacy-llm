# spacy-llm: Integrating LLMs into structured NLP pipelines

This package supports integration of Large Language Models (LLMs) into [spaCy](https://spacy.io/).
It adds an `llm` pipeline component to spaCy, allowing to prompt LLMs as part of your spaCy pipeline.
`llm` behaves like any other pipeline component and is (de-)serializable.

Each `llm` component is defined by two main settings:

- A [_Task_](api.md#Tasks), defining the prompt to send to the LLM as well as the functionality to parse the resulting response
  back into structured fields on spaCy's [Doc](https://spacy.io/api/doc) objects.
- A [_Backend_](api.md#Backends) defining the model to use and how to connect to it. Note that `spacy-llm` supports both access to external
  APIs (such as OpenAI) as well as access to self-hosted open-source LLMs (such as using Dolly through HuggingFace).

`spacy-llm` facilitates working with arbitrary prompting tools or libraries. Out of the box the following are supported:
- [`MiniChain`](https://github.com/srush/MiniChain)
- [`LangChain`](https://github.com/hwchase17/langchain)
- Access to GPT3 models from the [`OpenAI` API](https://platform.openai.com/docs/api-reference/introduction) via a simple default REST API.
- Access to the open-source [Dolly](https://huggingface.co/databricks) models hosted on HuggingFace.

The modularity of this repository allows you to easily implement your own functions, register them to the [spaCy registry](https://spacy.io/api/top-level#registry), 
and use them in a config file to power your NLP pipeline.

## ⏳ Install

`spacy-llm` will be installed automatically from spaCy v.3.5.3 onwards. For older spaCy v3 versions, you can run

```bash
python -m pip install spacy-llm
```

in the same virtual environment where you already have `spacy` [installed](https://spacy.io/usage).

## 🐍 Usage

The task and the backend have to be supplied to the `llm` pipeline component using [spaCy's config
system](https://spacy.io/api/data-formats#config). This package provides various built-in
functionality, as detailed in the [API](api.md) documentation.

### Example 1: run NER using an open-source model through HuggingFace

To run this example, ensure that you have a GPU enabled, and `transformers`, `torch` and CUDA installed.
For more background information, see the [DollyHF](api.md#spacydollyhfv1) section.

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
For more background information, see the [OpenAI](api.md#OpenAI) section.

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

## 📓 API

The details of all registered functions of this package can be found in the [API documentation](api.md).

Feel free to browse the source code to get inspiration on how to implement your own 
[task](https://github.com/explosion/spacy-llm/tree/main/spacy_llm/tasks) or 
[backend](https://github.com/explosion/spacy-llm/tree/main/spacy_llm/backends)!

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
