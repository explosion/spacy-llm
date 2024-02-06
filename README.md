<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>
<a href="https://explosion.ai"><img src="assets/logo.png" width="125" height="125" align="left" style="margin-right:30px" /></a>

<h1 align="center">
<span style="font: bold 38pt'Courier New';">spacy-llm</span>
<br>Structured NLP with LLMs
</h1>
<br><br>

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/explosion/spacy-llm/test.yml?branch=main)](https://github.com/explosion/spacy-llm/actions/workflows/test.yml)
[![pypi Version](https://img.shields.io/pypi/v/spacy-llm.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spacy-llm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

This package integrates Large Language Models (LLMs) into [spaCy](https://spacy.io), featuring a modular system for **fast prototyping** and **prompting**, and turning unstructured responses into **robust outputs** for various NLP tasks, **no training data** required.

## Feature Highlight

- Serializable `llm` **component** to integrate prompts into your spaCy pipeline
- **Modular functions** to define the [**task**](https://spacy.io/api/large-language-models#tasks) (prompting and parsing) and [**model**](https://spacy.io/api/large-language-models#models)
- Interfaces with the APIs of
  - **[OpenAI](https://platform.openai.com/docs/api-reference/)**
  - **[Cohere](https://docs.cohere.com/reference/generate)**
  - **[Anthropic](https://docs.anthropic.com/claude/reference/)**
  - **[Google PaLM](https://ai.google/discover/palm2/)**
  - **[Microsoft Azure AI](https://azure.microsoft.com/en-us/solutions/ai)**
- Supports open-source LLMs hosted on Hugging Face 🤗:
  - **[Falcon](https://huggingface.co/tiiuae)**
  - **[Dolly](https://huggingface.co/databricks)**
  - **[Llama 2](https://huggingface.co/meta-llama)**
  - **[OpenLLaMA](https://huggingface.co/openlm-research)**
  - **[StableLM](https://huggingface.co/stabilityai)**
  - **[Mistral](https://huggingface.co/mistralai)**
- Integration with [LangChain](https://github.com/hwchase17/langchain) 🦜️🔗 - all `langchain` models and features can be used in `spacy-llm`
- Tasks available out of the box:
  - [Named Entity Recognition](https://spacy.io/api/large-language-models#ner)
  - [Text classification](https://spacy.io/api/large-language-models#textcat)
  - [Lemmatization](https://spacy.io/api/large-language-models#lemma)
  - [Relationship extraction](https://spacy.io/api/large-language-models#rel)
  - [Sentiment analysis](https://spacy.io/api/large-language-models#sentiment)
  - [Span categorization](https://spacy.io/api/large-language-models#spancat)
  - [Summarization](https://spacy.io/api/large-language-models#summarization)
  - [Entity linking](https://spacy.io/api/large-language-models#nel)
  - [Translation](https://spacy.io/api/large-language-models#translation)
  - [Raw prompt execution for maximum flexibility](https://spacy.io/api/large-language-models#raw)
  - Soon:
    - Semantic role labeling
- Easy implementation of **your own functions** via [spaCy's registry](https://spacy.io/api/top-level#registry) for custom prompting, parsing and model integrations. For an example, see [here](https://spacy.io/usage/large-language-models#example-4).
- [Map-reduce approach](https://spacy.io/api/large-language-models#task-sharding) for splitting prompts too long for LLM's context window and fusing the results back together

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

## 🐍 Quickstart

Let's run some text classification using a GPT model from OpenAI. 

Create a new API key from openai.com or fetch an existing one, and ensure the
keys are set as environmental variables. For more background information, see
the documentation around setting [API keys](https://spacy.io/api/large-language-models#api-keys).

### In Python code

To do some quick experiments, from 0.5.0 onwards you can run:

```python
import spacy

nlp = spacy.blank("en")
llm = nlp.add_pipe("llm_textcat")
llm.add_label("INSULT")
llm.add_label("COMPLIMENT")
doc = nlp("You look gorgeous!")
print(doc.cats)
# {"COMPLIMENT": 1.0, "INSULT": 0.0}
```

By using the `llm_textcat` factory, the latest version of the built-in textcat task is used, 
as well as the default GPT-3-5 model from OpenAI.

### Using a config file

To control the various parameters of the `llm` pipeline, we can use 
[spaCy's config system](https://spacy.io/api/data-formats#config).
To start, create a config file `config.cfg` containing at least the following (or see the
full example
[here](https://github.com/explosion/spacy-llm/tree/main/usage_examples/textcat_openai)):

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.TextCat.v3"
labels = ["COMPLIMENT", "INSULT"]

[components.llm.model]
@llm_models = "spacy.GPT-4.v2"
```

Now run:

```python
from spacy_llm.util import assemble

nlp = assemble("config.cfg")
doc = nlp("You look gorgeous!")
print(doc.cats)
# {"COMPLIMENT": 1.0, "INSULT": 0.0}
```

That's it! There's a lot of other features - prompt templating, more tasks, logging etc. For more information on how to
use those, check out https://spacy.io/api/large-language-models.


## 🚀 Ongoing work

In the near future, we will

- Add more example tasks
- Support a broader range of models
- Provide more example use-cases and tutorials

PRs are always welcome!

## 📝️ Reporting issues

If you have questions regarding the usage of `spacy-llm`, or want to give us feedback after giving it a spin, please use
the [discussion board](https://github.com/explosion/spacy-llm/discussions).
Bug reports can be filed on the [spaCy issue tracker](https://github.com/explosion/spacy-llm/issues). Thank you!

## Migration guides

Please refer to our [migration guide](migration_guide.md).
