from typing import Callable, Optional, Iterable

import spacy
from spacy.tokens import Doc


@spacy.registry.llm("spacy.parse.Dummy.v1")
def dummy_parse() -> Callable[
    [Iterable[Doc], Iterable[str], Optional[str]], Iterable[Doc]
]:
    """Returns Callable parsing LLM responses and updating Doc instances with the extracted information.
    RETURNS (Callable[[Iterable[Doc], Iterable[str], Optional[str]], Iterable[Doc]]): Callable parsing LLM responses and
        mapping them to Doc instances.
    """

    def prompt_parse(
        docs: Iterable[Doc],
        prompt_responses: Iterable[str],
        response_field: Optional[str],
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            if response_field:
                setattr(doc._, response_field, prompt_response)

        return docs

    return prompt_parse
