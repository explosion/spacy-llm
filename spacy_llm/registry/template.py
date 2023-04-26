from typing import Callable, Iterable

import spacy
from spacy.tokens import Doc


@spacy.registry.llm("spacy.template.NoOp.v1")
def noop_template() -> Callable[[Iterable[Doc]], Iterable[str]]:
    """Returns Callable injecting Doc data into a prompt template and returning one fully specified prompt per passed
    Doc instance.
    RETURNS (Callable[[Iterable[Doc]], Iterable[str]]): Callable producing prompt strings.
    """
    template = ""

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        return [template]

    return prompt_template
