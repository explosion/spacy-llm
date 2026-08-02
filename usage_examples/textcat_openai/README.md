# Using GPT models from OpenAI

This example shows how you can use a model from OpenAI for categorizing texts in
zero- or few-shot settings. Here, we perform binary text classification to
determine if a given text is an `INSULT` or a `COMPLIMENT`.

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
python run_pipeline.py [TEXT] [PATH TO CONFIG] [PATH TO FILE WITH EXAMPLES]
```

For example:

```sh
python run_pipeline.py "You look great today! Nice shirt!" ./zeroshot.cfg
```
or, for few-shot:
```sh
python run_pipeline.py "You look great today! Nice shirt!" ./fewshot.cfg ./examples.jsonl
```

You can also include examples to perform few-shot annotation. To do so, use the 
`fewshot.cfg` file instead. You can find the few-shot examples in
the `examples.jsonl` file. Feel free to change and update it to your liking.
We also support other file formats, including `.yml`, `.yaml` and `.json`.


## OpenAI-compatible gateways

The OpenAI REST models accept an optional `endpoint` argument (full chat-completions URL).
You can point this at any OpenAI-compatible server — local (vLLM/Ollama) or a multi-model
gateway — and keep the same config shape:

```ini
[components.llm.model]
@llm_models = "spacy.GPT-4.v3"
name = "gpt-4o-mini"   # must be a model id the endpoint actually serves
endpoint = "https://api.daoxe.com/v1/chat/completions"
```

Set `OPENAI_API_KEY` to a key **issued by that endpoint**. Example multi-model gateway:
[DaoXE](https://daoxe.com) (`https://api.daoxe.com/v1`). Prefer HTTPS for remote endpoints.
