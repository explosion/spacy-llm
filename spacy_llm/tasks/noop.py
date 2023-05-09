from typing import Iterable

from spacy.tokens import Doc

from ..registry import registry


@registry.llm_tasks("spacy.NoOp.v1")
class NoopTask:
    def __init__(self):
        """Returns (1) templating Callable (injecting Doc data into a prompt template and returning one fully specified
        prompt per passed Doc instance) and (2) parsing callable (parsing LLM responses and updating Doc instances with the
        extracted information).
        RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
        """
        pass

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield "Don't do anything."

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        # Not doing anything
        return docs
