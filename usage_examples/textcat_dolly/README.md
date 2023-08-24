# Using open-source Dolly models hosted on Huggingface

This example shows how you can use the [open-source Dolly
models](https://github.com/databrickslabs/dolly) hosted on Huggingface for categorizing texts in
zero- or few-shot settings. Here, we perform binary text classification to
determine if a given text is an `INSULT` or a `COMPLIMENT`.

You can run the pipeline on a sample text via:

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

Finally, you can update the Dolly model in the configuration file. We're using
[`dolly-v2-3b`](https://huggingface.co/databricks/dolly-v2-3b) by default, but
you can change it to a larger model size like
[`dolly-v2-7b`](https://huggingface.co/databricks/dolly-v2-7b) or
[`dolly-v2-12b`](https://huggingface.co/databricks/dolly-v2-12b).
