# Linking entities with LLMs

This example shows how you can perform entity linking with LLMs.
This requires detecting named entities (i. e. performing NER) beforehand. You can do this using spaCy's `ner` 
component or `spacy-llm`'s NER task. The default config in this example utilizes the pretrained NER component from 
`en_core_web_md` for that.

> ⚠️ Ensure `en_core_web_md` is installed (`spacy download en_core_web_md`) before running this example.

Note that linking entities requires a knowledge base that defines the unique identifiers. `spacy-llm` natively supports spaCy's knowledge base class, but 
this object can contain any arbitrary knowledge base as long as the required interface is implemented.
For this example we provide a toy KB that supports a very limited number of entities (see 
[`el_kb_data.yml`](el_kb_data.yml)) - entities not listed in this file won't be linked.

First, create a new API key from [openai.com](https://platform.openai.com/account/api-keys) or fetch an existing one. Record the secret key and make sure this is
available as an environmental variable:

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
python run_pipeline.py \
    "The city of New York where John lives, lies in the state of New York." \
    ./zeroshot.cfg
```
or, for few-shot:
```sh
python run_pipeline.py \
    "The city of New York where John lives, lies in the state of New York." \
    ./fewshot.cfg \
    ./examples.yml
```

You can also include examples to perform few-shot annotation. To do so, use the
`fewshot.cfg` file instead. You can find the few-shot examples in
the `examples.yml` file. Feel free to change and update it to your liking.
We also support other file formats, including `.yaml`, `.jsonl` and `.json`.
