from typing import Callable, Iterable, Tuple

import spacy
from spacy.tokens import Doc


@spacy.registry.llm_tasks("spacy.NoOp.v1")
def noop_task() -> Tuple[
    Callable[[Iterable[Doc]], Iterable[str]],
    Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]],
]:
    """Returns (1) templating Callable (injecting Doc data into a prompt template and returning one fully specified
    prompt per passed Doc instance) and (2) parsing callable (parsing LLM responses and updating Doc instances with the
    extracted information).
    RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
    """
    template = "Don't do anything."

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield template

    def prompt_parse(
        docs: Iterable[Doc],
        prompt_responses: Iterable[str],
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            pass

        return docs

    return prompt_template, prompt_parse
