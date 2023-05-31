# Using GPT models from OpenAI

This example shows how you can use a model from OpenAI to recognize named entities using the MiniChain backend.

First, create a new API key from
[openai.com](https://platform.openai.com/account/api-keys) or fetch an existing
one. Record the secret key and make sure this is available as an environmental
variable:

```sh
export OPENAI_API_KEY="sk-..."
export OPENAI_API_ORG="org-..."
```

Then, you can run the pipeline on a sample text via:

```sh
python run_pipeline.py [TEXT] [PATH TO CONFIG]
```

For example:

```sh
python run_pipeline.py "Jack and Jill went up the hill." ./ner.cfg
```
