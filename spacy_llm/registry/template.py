from typing import Callable, Iterable

import spacy
from spacy.tokens import Doc


@spacy.registry.llm("spacy.template.Dummy.v1")
def dummy_template() -> Callable[[Iterable[Doc]], Iterable[str]]:
    """Returns Callable injecting Doc data into a prompt template and returning one fully specified prompt per passed
    Doc instance.
    RETURNS (Callable[[Iterable[Doc]], Iterable[str]]): Callable producing prompt strings.
    """
    template = "What is {value} times three? Respond with the exact number."

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        return [template.format(value=len(doc)) for doc in docs]

    return prompt_template
