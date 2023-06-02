# Relation extraction using LLMs

This example shows how you can use a model from OpenAI for relation extraction in
zero- and few-shot settings.

Here, we use the pretrained [`en_core_web_md` model](https://spacy.io/models/en#en_core_web_sm)
to perform Named Entity Recognition (NER) using a fast and properly evaluated pipeline.
Then, we leverage the OpenAI API to detect the relations between the extracted entities.
In this example, we focus on two simple relations: `LivesIn` and `Visits`.

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
python run_pipeline.py \
    "Laura just bought an apartment in Boston." \
    ./zeroshot.cfg
```

or, with few-shot:

```sh
python run_pipeline.py \
    "Laura just bought an apartment in Boston." \
    ./fewshot.cfg
    ./examples.jsonl
```

You can also include examples to perform few-shot annotation. To do so, use the
`openai_rel_fewshot.cfg` file instead. You can find the few-shot examples in
the `examples.jsonl` file. Feel free to change and update it to your liking.
We also support other file formats, including `.json`, `.yml` and `.yaml`.
