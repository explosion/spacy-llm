# spacy-llm
This package supports integration of LLM APIs into spaCy. It adds an `llm` pipeline component to spaCy, allowing to prompt 
LLMs as part of your spaCy pipeline. `llm` behaves like any other pipeline component and is (de-)serializable. 
Self-hosted LLMs (e. g. Dolly) are not supported yet, but are on our roadmap.

`spacy-llm` assumes three functionalities to be implemented for a use case:
- _Templating_, i. e. defining a prompt template and injecting the relevant data from your `Doc` instances into this 
  template to generate fully formed prompts.
- _API access and prompting_, i. e. connecting to the LLM API and executing the prompt. A minimal wrapper layer for 
   compatibility is provided, but you are free to use whatever backend (`langchain`, `minichain`, a hand-rolled backend 
   connecting to the API of your choice,  ...) you prefer for connecting to the LLM API and executing the prompt(s) 
   underneath.
- _Parsing_, i. e. parsing the LLM response(s), extracting the useful bits from it and mapping these back onto your 
  documents.

As parsing an LLM response very much depends on how the prompt for this response looked like, templating and parsing are
packaged together as _tasks_.

## 🖊️ Usage

### Configuration

The code for templating, prompting and parsing has to be supplied to the `llm` pipeline component using spaCy's config 
system. The default configuration is as follows:
```ini
[components.llm] 
factory = "llm"
# Factory function for Callable returning the templating and parsing functions. 
task = {"@llm_tasks": "spacy.NoOp.v1"}
# Factory function for Callable generating instance of backend to use. In this case: the MiniChain wrapper that is already 
# implemented. This corresponds to the "prompting" step and includes managing the connection to the LLM API.
backend = {
    "api": "OpenAI",
    "@llm_backends": "spacy.MiniChain.v1",
    "config": {}
}
```

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

### Authentication

This package does not explicitly implement any kind of credential management or storage. Please refer to the 
documentation of the libraries you are using to find out how to set your credentials.

## 🔨 Supported tools

`spacy-llm` facilitates working with arbitrary prompting tools or libraries. Out of the box the following are supported:
- [`MiniChain`](https://github.com/srush/MiniChain)
- [`LangChain`](https://github.com/hwchase17/langchain)

## ⚙️ Supported use cases

- TODO

## ⚠️ Warning: experimental package

This package is experimental and it is possible that changes made to the interface will be breaking in minor version updates.

## ⏳ Install

```bash
pip install spacy-llm
```

## 📝️ Reporting issues

Please report all issues with `spacy-llm` in the [spaCy issue tracker](https://github.com/explosion/spaCy/issues). If
you have questions regarding the usage of `spacy-llm`, use the 
[discussion board](https://github.com/explosion/spaCy/discussions). Thank you! 