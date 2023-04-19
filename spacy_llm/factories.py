from typing import Callable, Optional, Iterable

import minichain
import spacy
from spacy.tokens import Doc

from .api import Promptable, PromptableMiniChain

# Create new registry.
if "llm" not in spacy.registry.get_registry_names():
    spacy.registry.create("llm", entry_points=True)


@spacy.registry.llm("spacy.api.MiniChain.v1")
def api_minichain(
    backend: str, prompt: Callable[[minichain.Backend, Iterable[str]], Iterable[str]]
) -> Callable[[], Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    prompt (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]): Callable executing prompts.
    RETURNS (Promptable): Promptable wrapper for Minichain.
    """

    def init() -> Promptable:
        return PromptableMiniChain(backend, prompt)

    return init


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


@spacy.registry.llm("spacy.prompt.MiniChainSimple.v1")
def minichain_simple_prompt() -> Callable[
    [minichain.Backend, Iterable[str]], Iterable[str]
]:
    """Returns Promptable wrapper for Minichain.
    RETURNS (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]:): Callable executing simple prompts on a
        MiniChain backend.
    """

    def prompt(backend: minichain.Backend, prompts: Iterable[str]) -> Iterable[str]:
        @minichain.prompt(backend())
        def _prompt(model: minichain.backend, prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt


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
