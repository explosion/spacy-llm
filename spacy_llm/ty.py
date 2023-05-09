from typing import Callable, Iterable, TypeVar

from spacy.tokens import Doc

from .compat import Protocol, runtime_checkable


_Prompt = TypeVar("_Prompt")
_Response = TypeVar("_Response")

PromptExecutor = Callable[[Iterable[_Prompt]], Iterable[_Response]]


@runtime_checkable
class LLMTask(Protocol):
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[_Prompt]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[_Response]
    ) -> Iterable[Doc]:
        ...
