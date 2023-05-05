from typing import Any, Iterable

from spacy.tokens import Doc

from .compat import Protocol, runtime_checkable


@runtime_checkable
class LLMTask(Protocol):
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[Any]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[Any]
    ) -> Iterable[Doc]:
        ...
