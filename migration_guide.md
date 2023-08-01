# Migration guides

<details open>
  <summary>0.3.x to 0.4.x</summary>

## `0.3.x` to `0.4.x`

`0.4.x` significantly refactors the code to make it more robust and the config more intuitive. 0.4.0 changes the config 
paradigm from `backend`- to `model`-centric. This is reflected in the external API in a different config structure.

Remember that there are three different types of models: the first uses the native REST implementation to communicate
with hosted LLMs, the second builds on HuggingFace's `transformers` model to run models locally and the third leverages
`langchain` to operate on hosted or local models. While the config for all three is rather similar (especially in 
0.4.x), there are differences in how these models have to be configured. We show how to migrate your config from 0.3.x
to 0.4.x for each of these model types.

### All model types 
- The registry name has changed - instead of `@llm_backends`, use `@llm_models`.
- The `api` attribute has been removed.

### Models using REST

This is the default method to communicate with hosted models. Whenever you don't explicitly use LangChain models
(see section at the bottom) or run models locally, you are using this kind of model.

In `0.3.x`:
```ini
[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "gpt-3.5-turbo", "temperature": 0.3}
```
In `0.4.x`:
```ini
[components.llm.model]
@llm_models = "spacy.GPT-3-5.v1"
name = "gpt-3.5-turbo"
config = {"temperature": 0.3}
```
Note that the factory function (marked with `@`) refers to the name of the model. Variants of the same model can be 
specified with the `name` attribute - for `gpt-3.5` this could be `"gpt-3.5-turbo"` or `"gpt-3.5-turbo-16k"`.

### Models using HuggingFace

On top of the changes described in the section above, HF models like `spacy.Dolly.v1` now accept `config_init` and 
`config_run` to reflect that differerent arguments can be passed at init or run time.

In `0.3.x`:
```ini
[components.llm.backend]
@llm_backends = "spacy.Dolly_HF.v1"
model = "databricks/dolly-v2-3b"
config = {}
```
In `0.4.x`:
```ini
[components.llm.model]
@llm_models = "spacy.Dolly.v1"
name = "dolly-v2-3b"  # or databricks/dolly-v2-3b - the prefix is optional
config_init = {}  # Arguments passed to HF model at initialization time
config_run = {}  # Arguments passed to HF model at inference time 
```

### Models using LangChain

LangChain models are now accessible via `langchain.[API].[version]`, e. g. `langchain.OpenAI.v1`. Other than that the
changes from 0.3.x to 0.4.x are identical with REST-based models.

In `0.3.x`:
```ini
[components.llm.backend]
@llm_backends = "spacy.LangChain.v1"
api = "OpenAI"
config = {"temperature": 0.3}
```

In `0.4.x`:
```ini
[components.llm.model]
@llm_models = "langchain.OpenAI.v1"
name = "gpt-3.5-turbo"
config = {"temperature": 0.3}
```

</details>