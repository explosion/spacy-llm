from typing import Callable, Iterable

from spacy.tokens import Doc

from .util import registry


@registry.templates("spacy-llm.NoOp.v1")
def noop_template() -> Callable[[Iterable[Doc]], Iterable[str]]:
    """Returns Callable injecting Doc data into a prompt template and returning one fully specified prompt per passed
    Doc instance.
    RETURNS (Callable[[Iterable[Doc]], Iterable[str]]): Callable producing prompt strings.
    """
    template = "Don't do anything."

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield template

    return prompt_template
