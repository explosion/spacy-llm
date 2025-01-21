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
python run_pipeline.py [TEXT] [PATH TO CONFIG] [PATH TO FILE WITH EXAMPLES]
```

For example:

```sh
python run_pipeline.py \
    "Laura bought an apartment last month in Berlin." \
    ./zeroshot.cfg
```
or, for few-shot:
```sh
python run_pipeline.py \
    "Laura bought an apartment last month in Berlin." \
    ./fewshot.cfg \
    ./examples.jsonl
```

LLM-response:
```sh
LLM response for doc: Laura bought an apartment last month in Boston.

Step 1: Extract the Predicates for the Text
Predicates: bought

Step 2: For each Predicate, extract the Semantic Roles in 'Text'
Text: Laura bought an apartment last month in Boston.
Predicate: bought
ARG-0: Laura
ARG-1: an apartment
ARG-2: 
ARG-M-TMP: last month
ARG-M-LOC: in Boston
```
std output:
```sh
Text: Laura bought an apartment last month in Boston.
SRL Output:
Predicates: ['bought']
Relations: [('bought', [('ARG-0', 'Laura'), ('ARG-1', 'an apartment'), ('ARG-M-TMP', 'last month'), ('ARG-M-LOC', 'in Boston')])]
```