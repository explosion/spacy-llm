from typing import Iterable

from spacy.tokens import Doc

from ..registry import registry

NOOP_PROMPT = "Don't do anything."


@registry.llm_tasks("spacy.NoOp.v1")
class NoopTask:
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for _ in docs:
            yield NOOP_PROMPT

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        # Not doing anything
        return docs
