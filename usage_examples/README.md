# Usage examples

This directory contains different examples on how you can use `spacy-llm` for
common NLP tasks. Each directory will contain a sample configuration and an
optional `examples.yml` file for few-shot annotation.

## The configuration file

Each configuration file contains an `llm` component that takes in a `task` and a
`backend` as its parameters. The `task` defines how the prompt is structured and
how the corresponding LLM output will be parsed whereas the `backend` defines
which model to use and how to connect to it.

```
...
[components]

[components.llm]
factory = "llm"

# Defines the prompt you'll send to an LLM, and how the corresponding output 
# will be parsed.
[components.llm.task]
...

# Defines which model to use (open-source or third-party API) and how to connect
# to it (e.g., REST, MiniChain, LangChain).
[components.llm.backend]
...
```

The configuration files are based on [spaCy's configuration
system](https://spacy.io/api/data-formats#config). This means that `spacy-llm`
is modular and it's easy enough to implement your own tasks.

## Writing your own task

The common use-case for `spacy-llm` is to use a large language model (LLM) to
power a natural language processing pipeline. In `spacy-llm`, we define these
actions as **tasks**.

Think of a `task` as something you want an LLM to do. In our examples, we ask
an LLM to find named-entities or categorize a text. Note that an LLM's output
must be contained in a spaCy [`Doc`](https://spacy.io/api/doc). For example,
named-entities are stored in [`doc.ents`](https://spacy.io/api/doc#ents) while
text categorization results are in [`doc.cats`](https://spacy.io/api/doc#cats).

To write a `task`, you need to implement two functions:

- **generate_prompts(docs: Iterable[Doc]) -> Iterable[str]**: a function that
takes in a list of spaCy [`Doc`](https://spacy.io/api/doc) objects and transform
them into a list of prompts. These prompts will then be sent to the LLM in the
`backend`.

- **parse_responses(docs: Iterable[Doc], responses: Iterable[str]) ->
Iterable[Doc]**: a function for parsing the LLM's outputs into spaCy
[`Doc`](https://spacy.io/api/doc) objects. You also have access to the input
`Doc` objects so you can store the outputs into one of its attributes.

The `spacy-llm` library requires tasks to be defined as a class and registered in the `llm_tasks` registry:

```python
from spacy_llm.registry import registry

@registry.llm_tasks("spacy.MyTask.v1")
class MyTask:
    def __init__(self, labels: str):
        ...

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        ...
```

(and in the config)

```
...
[components.llm.task]
@llm_tasks = "spacy.MyTask.v1"
labels = LABEL1,LABEL2,LABEL3
...
```

You can check sample tasks for named-entity recognition and text categorization
in the `spacy_llm/tasks/` directory. We also recommend checking out the
`spacy.NoOp.v1` task for a barebones implementation to pattern your task from.

<!-- TODO 

### Writing your own backend

-->
