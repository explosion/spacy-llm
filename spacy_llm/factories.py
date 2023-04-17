from typing import Callable, Optional, Iterable

import spacy
from spacy.tokens import Doc

from .api import Promptable, PromptableMiniChain

# Create new registry.
if "llm" not in spacy.registry.get_registry_names():
    spacy.registry.create("llm", entry_points=True)


@spacy.registry.llm("spacy.API.MiniChain.v1")
def api_minichain(backend: str) -> Callable[[], Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    RETURNS (Promptable): Promptable wrapper for Minichain.
    """

    def init() -> Promptable:
        return PromptableMiniChain(backend)

    return init


@spacy.registry.llm("spacy.DummyTemplate.v1")
def dummy_template() -> Callable[[Iterable[Doc]], Iterable[str]]:
    """Returns Callable injecting Doc data into a prompt template and returning one fully specified prompt per passed
    Doc instance.
    RETURNS (Callable[[Iterable[Doc]], Iterable[str]]): Callable producing prompt strings.
    """
    template = "What is {value} times three? Respond with the exact number."

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        return [template.format(value=len(doc)) for doc in docs]

    return prompt_template


@spacy.registry.llm("spacy.DummyParse.v1")
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
