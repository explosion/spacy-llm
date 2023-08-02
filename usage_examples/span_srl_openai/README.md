# Semantic Role Labeling (SRL) using LLMs

This example shows how you can use a model from OpenAI for SRL in
zero- and few-shot settings.


We leverage the OpenAI API to detect the predicates and argument roles in a sentence.
In the example below, we focus on the predicate "bought" and ARG-0, ARG-1, and ARG-M-LOC.

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
    "Laura bought an apartment in Boston last month." \
    ./zeroshot.cfg
```
Output:
```shell
Text: Laura bought an apartment last month in Boston.
SRL Output:
Predicates: ['bought']
Relations: [('bought', [('ARG-0', 'Laura'), ('ARG-1', 'an apartment'), ('ARG-M-TMP', 'last month'), ('ARG-M-LOC', 'in Boston')])]
```