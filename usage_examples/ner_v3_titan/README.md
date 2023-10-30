# Using Titan Express Model from Amazon Bedrock for Named Entity Recognition (NER)


This example shows how you can use a model from OpenAI for Named Entity Recognition (NER).
The NER prompt is based on the [PromptNER](https://arxiv.org/abs/2305.15444) paper and
utilizes Chain-of-Thought reasoning to extract named entities.

First, create a new credentials from AWS Console
Record the secret key and make sure this is available as an environmental
variable:

```sh
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_SESSION_TOKEN=""
```

Then, you can run the pipeline on a sample text via:


```sh
python run_pipeline.py [TEXT] [PATH TO CONFIG] [PATH TO FILE WITH EXAMPLES]
```

For example:

```sh
python run_pipeline.py \
    ""Sriracha sauce goes really well with hoisin stir fry, but you should add it after you use the wok." \
    ./fewshot.cfg
    ./examples.json
```

This example assings labels for DISH, INGREDIENT, and EQUIPMENT.

You can change around the labels and examples for your use case.
You can find the few-shot examples in the
`examples.json` file. Feel free to change and update it to your liking.
We also support other file formats, including `yml` and `jsonl` for these examples.


### Negative examples

While not required, The Chain-of-Thought reasoning for the `spacy.NER.v3` task
works best in our experience when both positive and negative examples are provided.

This prompts the Language model with concrete examples of what **is not** an entity
for your use case.

Here's an example that helps define the INGREDIENT label for the LLM.

```json
[
    {
        "text": "You can't get a great chocolate flavor with carob.",
        "spans": [
            {
                "text": "chocolate",
                "is_entity": false,
                "label": "==NONE==",
                "reason": "is a flavor in this context, not an ingredient"
            },
            {
                "text": "carob",
                "is_entity": true,
                "label": "INGREDIENT",
                "reason": "is an ingredient to add chocolate flavor"
            }
        ]
    }
    ...
]
```

In this example, "chocolate" is not an ingredient even though it could be in other contexts.
We explain that via the "reason" property of this example.
