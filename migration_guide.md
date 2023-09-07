# Migration guides

<details open>
  <summary>0.4.x to 0.5.x</summary>

## `0.4.x` to `0.5.x`

`0.5.x` includes internal refactoring that should have minimal to zero impact to the user experience. Mostly, configurations from `0.4.x` 
should just work on `0.5.x`.

### Use the new Chain-of-Thought NER prompting

We've implemented Chain-of-Thought (CoT) prompting for SpanCat and NER tasks, 
based on the
[PromptNER paper](https://arxiv.org/pdf/2305.15444.pdf) by Ashok and Lipton
(2023).

This implementation is available as `spacy.SpanCat.v3` and `spacy.NER.v3`. 
Zero-shot prompting should remain pretty much the same, though behind the scenes, 
a dummy prompt example will be used for the CoT implementations. 

For few-shot learning, the provided examples need to be provided in a slightly 
[different format](https://spacy.io/api/large-language-models#ner) than the v1 and v2 versions.

First, you can provide an explicit `description` of what entities should look like. 

In `0.4.x`:
```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = ["DISH", "INGREDIENT", "EQUIPMENT"]
```

In `0.5.x`:
```ini
[components.llm.task]
@llm_tasks = "spacy.NER.v3"
labels = ["DISH", "INGREDIENT", "EQUIPMENT"]
description = Entities are the names food dishes,
    ingredients, and any kind of cooking equipment.
    Adjectives, verbs, adverbs are not entities.
    Pronouns are not entities.
```

Further, the examples for few-shot learning also look different, and you can include both positive as well as negative examples 
using the new fields `is_entity` and `reason`.

In `0.4.x`:
```json
[
  {
    "text": "You can't get a great chocolate flavor with carob.",
    "entities": {
      "INGREDIENT": ["carob"]
    }
  },
    ...
]
```

In `0.5.x`:
```json
[
    {
        "text": "You can't get a great chocolate flavor with carob.",
        "spans": [
            {
                "text": "chocolate",
                "is_entity": false,
                "label": "==NONE==",
                "reason": "is a flavor in this context, not an ingredient"
            },
            {
                "text": "carob",
                "is_entity": true,
                "label": "INGREDIENT",
                "reason": "is an ingredient to add chocolate flavor"
            }
        ]
    },
    ...
]
```

For a full example using 0.5.0 with Chain-of-Thought prompting for NER, see 
[this usage example](https://github.com/explosion/spacy-llm/tree/main/usage_examples/ner_v3_openai).

</details>

<details>
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