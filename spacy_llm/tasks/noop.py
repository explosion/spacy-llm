from typing import Iterable

from spacy.tokens import Doc

from ..registry import registry

_NOOP_PROMPT = "Don't do anything."


@registry.llm_tasks("spacy.NoOp.v1")
def make_noop_task():
    return NoopTask()


class NoopTask:
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for _ in docs:
            yield _NOOP_PROMPT

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        # Not doing anything
        return docs

    @property
    def prompt_template(self) -> str:
        return """
        This is the NoOp
        prompt template
        """
