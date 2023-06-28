# Usage examples

This directory contains different examples on how you can use `spacy-llm` to
simulate or prototype common NLP tasks. Each directory contains a sample
configuration and an optional `examples.yml` file for few-shot annotation.

## The configuration file

Each configuration file contains an `llm` component that takes in a `task` and a
`model` as its parameters. `task` defines how the prompt is structured and
how the corresponding LLM output will be parsed whereas `model` defines
which model to use and how to connect to it.

```ini
...
[components]

[components.llm]
factory = "llm"

# Defines the prompt you'll send to an LLM, and how the corresponding output
# will be parsed.
[components.llm.task]
...

# Defines which model to use (open-source or third-party API) and how to connect
# to it (e.g., REST, LangChain, locally via HuggingFace, ...).
[components.llm.model]
...
```

The configuration files are based on [spaCy's configuration
system](https://spacy.io/api/data-formats#config). This means that `spacy-llm`
is modular and it's easy to implement your own tasks.

## Writing your own task

The common use-case for `spacy-llm` is to use a large language model (LLM) to
power a natural language processing pipeline. In `spacy-llm`, we define these
actions as **tasks**.

Think of a `task` as something you want an LLM to do. In our examples, we ask an
LLM to find named entities or categorize a text. Note that an LLM's output
should eventually be stored in a spaCy [`Doc`](https://spacy.io/api/doc). For
example, named entities are stored in
[`doc.ents`](https://spacy.io/api/doc#ents) while text categorization results
are in [`doc.cats`](https://spacy.io/api/doc#cats).

To write a
[`task`](https://github.com/explosion/spacy-llm/blob/main/README.md#tasks), you
need to implement two functions:

- **`generate_prompts(docs: Iterable[Doc]) -> Iterable[str]`**: a function that
  takes in a list of spaCy [`Doc`](https://spacy.io/api/doc) objects and transforms
  them into a list of prompts. These prompts will then be sent to the LLM in the
  `model`.
- **`parse_responses(docs: Iterable[Doc], responses: Iterable[str]) -> Iterable[Doc]`**: a function for parsing the LLM's outputs into spaCy
  [`Doc`](https://spacy.io/api/doc) objects. You also have access to the input
  `Doc` objects so you can store the outputs into one of its attributes.

The `spacy-llm` library requires tasks to be defined as a class and registered in the `llm_tasks` registry:


```python
from spacy_llm.registry import registry
from spacy_llm.util import split_labels


@registry.llm_tasks("my_namespace.MyTask.v1")
def make_my_task(labels: str, my_other_config_val: float) -> "MyTask":
    labels_list = split_labels(labels)
    return MyTask(labels=labels_list, my_other_config_val=my_other_config_val)


class MyTask:
    def __init__(self, labels: List[str], my_other_config_val: float):
        ...

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        ...
```

```ini
# config.cfg (excerpt)
[components.llm.task]
@llm_tasks = "my_namespace.MyTask.v1"
labels = LABEL1,LABEL2,LABEL3
my_other_config_val = 0.3
```

You can check sample tasks for Named Entity Recognition and text categorization
in the `spacy_llm/tasks/` directory. We also recommend checking out the
`spacy.NoOp.v1` task for a barebones implementation to pattern your task from.

## Using LangChain

`spacy-llm` integrates [LangChain](https://github.com/hwchase17/langchain) to leverage its features for prompt management and LLM usage to allow users
to leverage their functionality in their spaCy workflows. A built-in example for this is 

An integrated third-party library can be used by configuring the `llm` component to use the respective model, e. g.:

```ini
[components.llm.model]
@llm_models = "langchain.OpenAI.v1"
name = "gpt-3.5-turbo"
```


<!-- The `usage_examples` directory contains example for all integrated third-party -->

## Writing your own model

In `spacy-llm`, the [**model**](../README.md#models) is responsible for the
interaction with the actual LLM model. The latter can be an
[API-based service](../README.md#spacyrestv1), or a local model - whether
you [downloaded it from the Hugging Face Hub](../README.md#spacydollyhfv1)
directly or finetuned it with proprietary data.

`spacy-llm` lets you implement your own custom model so you can try out the
latest LLM interface out there. Bear in mind that tasks are responsible for
creating the prompt and parsing the response – and both can be arbitrary objects.
Hence, a model's call signature should be consistent with that of the task you'd like it to run.

In other words, `spacy-llm` roughly performs the following pseudo-code behind the scenes:

```python
prompts = task.generate_prompts(docs)
responses = model(prompts)
docs = task.parse_responses(docs, responses)
```

Let's write a dummy model that provides a random output for the
[text classification task](../README.md#spacytextcatv1).

```python
from spacy_llm.registry import registry
import random
from typing import Iterable

@registry.llm_models("RandomClassification.v1")
def random_textcat(labels: str):
    labels = labels.split(",")
    def _classify(prompts: Iterable[str]) -> Iterable[str]:
        for _ in prompts:
            yield random.choice(labels)
    
    return _classify
```

```ini
...
[components.llm.task]
@llm_tasks = "spacy.TextCat.v1"
labels = LABEL1,LABEL2,LABEL3


[components.llm.model]
@llm_models = "RandomClassification.v1"
labels = ${components.llm.task.labels}  # Make sure to use the same label
...
```

Of course, this particular model is not very realistic
(it does not even interact with an actual LLM model!).
But it does show how you would go about writing custom
and arbitrary logic to interact with any LLM implementation.

Note that in all built-in tasks prompts and responses are expected to be of type `str`, while all built-in model
support `str` (or `Any`) types. All built-in tasks and models are therefore inter-operable. It's possible to work with 
arbitrary objects instead of `str` though - which might be useful if you want some third-party abstractions for prompts
or responses.
