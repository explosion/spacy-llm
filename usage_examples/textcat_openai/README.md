# Using GPT models from OpenAI

This example shows how you can use a model from OpenAI for categorizing texts in
zero- or few-shot settings. Here, we perform binary text classification to
determine if a given text is an `INSULT` or a `COMPLIMENT`.

First, create a new API key from
[openai.com](https://platform.openai.com/account/api-keys) or fetch an existing
one. Record the secret key and make sure this is available as an environmental
variable. Set them in a `.env` file in this directory:

```sh
OPENAI_API_KEY="sk-..."
```

Then, you can run the pipeline on a sample text via:

```sh
python run_textcat_openai_pipeline.py [TEXT] [PATH TO CONFIG]
```

For example:

```sh
python run_textcat_openai_pipeline.py \
    "You look great today! Nice shirt!" \
    ./openai_textcat_zeroshot.cfg
```

You can also include examples to perform few-shot annotation. To do so, use the 
`openai_textcat_fewshot.cfg` file instead. You can find the few-shot examples in
the `textcat_examples.jsonl` file. Feel free to change and update it to your liking.
We also support other file formats, including `.yml`, `.yaml` and `.json`.