## Using GPT-3.5 models from OpenAI

This example shows how you can use the [GPT-3.5 model from
OpenAI](https://platform.openai.com/docs/models/gpt-3-5) for categorizing texts
in zero- or few-shot settings. Here, we perform binary text classification to
determine if a given text is an `INSULT` or a `COMPLIMENT`.

First, create a new API key from
[openai.com](https://platform.openai.com/account/api-keys) or fetch an existing
one. Record the secret key and make sure this is available as an environmental
variable. Set them in a `.env` file in this directory:

```
OPENAI_API_KEY="sk-..."
```

Then, you can run the pipeline on a sample text via:

```sh
python run_textcat_openai_pipeline.py [TEXT] [PATH TO CONFIG]
```

For example:

```sh
python run_ner_dolly_pipeline.py \
    "One half of me is yours, the other half yours--Mine own, I would say; but if mine, then yours, and so all yours" \
    ./dolly_ner_zeroshot.cfg
```

You can also include examples to perform few-shot annotation. To do so, use the 
`openai_textcat_fewshot.cfg` file instead. You can find the few-shot examples in
the `textcat_examples.yml` file. Feel free to change and update it to your liking.