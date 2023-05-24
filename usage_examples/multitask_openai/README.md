# Performing multiple tasks in a single pipeline

This example shows how you can perform multiple LLM-backed tasks within
a single spaCy pipeline.

We could create a new custom task that performs all objectives in a single
LLM query, but in this example we'll only use built-in task templates to
see how easy it is to compose them. Note that breaking down tasks this way
might be a better choice anyway, since it allows you to better control the
performance of your pipeline.

This example shows how you can use a model from OpenAI for categorizing texts
as well as detect entities of interest in zero- or few-shot settings.
Here, we perform binary text classification to determine if a given text
is an `ORDER` or a `INFORMATION` request.

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
python run_pipeline.py \
    "I'd like to order a small margherita pizza" \
    ./zeroshot.cfg
```
or, for few-shot:
```sh
python run_pipeline.py \
    "I'd like to order a small margherita pizza" \
    ./fewshot.cfg \
    ./examples.yml
```

You can also include examples to perform few-shot annotation. To do so, use the
`fewshot.cfg` file instead. You can find the few-shot examples in
the `examples.yml` file. Feel free to change and update it to your liking.
We also support other file formats, including `.yaml`, `.jsonl` and `.json`.
