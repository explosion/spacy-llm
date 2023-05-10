# spacy-llm: integrating raw output from LLMs into structured NLP pipelines

This package supports integration of LLM APIs into spaCy. It adds an `llm` pipeline component to spaCy, allowing to prompt 
LLMs as part of your spaCy pipeline. `llm` behaves like any other pipeline component and is (de-)serializable. 
Each `llm` component is defined by two main settings:
- A _Task_, defining the prompt to send to the LLM as well as the functionality to parse the resulting response 
  back into structured fields on spaCy's [Doc](https://spacy.io/api/doc) objects.  
- A _Backend_ defining the model to use and how to connect it. Note that this package supports both access to external
  APIs (such as OpenAI) as well as local access to self-hosted open-source LLMs (such as using Dolly through HuggingFace).

## ⏳ Install

`spacy-llm` will be installed automatically from spaCy v.3.5.3 onwards. For older spaCy v3 versions, you can run

```bash
pip install spacy-llm
```


## 🖊️ Usage

The task and the backend have to be supplied to the `llm` pipeline component using [spaCy's config 
system](https://explosion.ai/blog/spacy-v3-project-config-systems). This package provides various built-in 
functionality, documented below in the [API](#📓-API) section.

### Example using an open-source model through HuggingFace

Add this to your config, or see the full example here:

```ini
[components.llm] 
factory = "llm"

[components.llm.task] 
@llm_tasks = "spacy.NER.v1"
labels = PER,ORG,LOC

[components.llm.backend]
@llm_backends = "spacy.DollyHF.v1"
model = "databricks/dolly-v2-3b"
```
Note that HuggingFace will download this model the first time you use it - you can 
[define the cached directory](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache) 
by setting the environmental variable `HF_HOME`. 
Also, you can upgrade the model to be "databricks/dolly-v2-12b" for better performance.

### Minimal example

The `llm` component behaves as any other spaCy component does, so adding it to an existing pipeline follows the same 
pattern:
```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("llm")
doc = nlp("This is a test")
```
Note however that this won't make much sense without configuring your `llm` component properly (see 
section above) - otherwise the default configuration is run, which runs a dummy prompt. 

For more details on spaCy's configuration system consult the [spaCy docs](https://spacy.io/api/data-formats#config).  

## ⚠️ Warning: experimental package

This package is experimental and it is possible that changes made to the interface will be breaking in minor version updates.

### Authentication

This package does not explicitly implement any kind of credential management or storage. Please refer to the 
documentation of the libraries you are using to find out how to set your credentials.





## 📝️ Reporting issues

Please report all issues with `spacy-llm` in the [spaCy issue tracker](https://github.com/explosion/spaCy/issues). If
you have questions regarding the usage of `spacy-llm`, use the 
[discussion board](https://github.com/explosion/spaCy/discussions). Thank you! 

## 📓 API

### Tasks

- _Templating_, i. e. defining a prompt template and injecting the relevant data from your `Doc` instances into this 
  template to generate fully formed prompts.

- _Parsing_, i. e. parsing the LLM response(s), extracting the useful bits from it and mapping these back onto your 
  documents.

### Backends

- _API access and prompting_, i. e. connecting to the LLM API and executing the prompt. A minimal wrapper layer for 
   compatibility is provided, but you are free to use whatever backend (`langchain`, `minichain`, a hand-rolled backend 
   connecting to the API of your choice,  ...) you prefer for connecting to the LLM API and executing the prompt(s) 
   underneath.
- 
`spacy-llm` facilitates working with arbitrary prompting tools or libraries. Out of the box the following are supported:
- [`MiniChain`](https://github.com/srush/MiniChain)
- [`LangChain`](https://github.com/hwchase17/langchain)