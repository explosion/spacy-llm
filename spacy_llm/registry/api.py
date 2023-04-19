from typing import Callable, Iterable

import minichain
import spacy

from ..api import Promptable, PromptableMiniChain


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
