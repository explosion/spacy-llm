from typing import Iterable

from confection import Config
from spacy.tokens import Doc

from spacy_llm.registry import registry
from spacy_llm.util import assemble_from_config

config = """[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"
save_io = True

[components.llm.task]
@llm_tasks = "my_namespace.MyTask.v1"

[components.llm.model]
@llm_models = "spacy.NoOp.v1"
"""


@registry.llm_tasks("my_namespace.MyTask.v1")
def make_my_task() -> "MyTask":
    return MyTask()


class MyTask:
    def __init__(self):
        self._template = "Do a sumersault"

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield self._template

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        return docs


def test_custom_task():
    nlp = assemble_from_config(Config().from_str(config))
    nlp("This is a test.")
