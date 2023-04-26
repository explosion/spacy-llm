from typing import Callable, Iterable

import spacy
from spacy.tokens import Doc


@spacy.registry.llm("spacy.parse.NoOp.v1")
def noop_parse() -> Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]]:
    """Returns Callable parsing LLM responses and updating Doc instances with the extracted information.
    RETURNS (Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]]): Callable parsing LLM responses and
        mapping them to Doc instances.
    """

    def prompt_parse(
        docs: Iterable[Doc],
        prompt_responses: Iterable[str],
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            pass

        return docs

    return prompt_parse
