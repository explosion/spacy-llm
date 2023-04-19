from typing import Callable, Iterable, Any

import langchain.llms
import minichain
import spacy

from .. import api


@spacy.registry.llm("spacy.api.MiniChain.v1")
def api_minichain(
    backend: str, prompt: Callable[[minichain.Backend, Iterable[Any]], Iterable[Any]]
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    prompt (Callable[[minichain.Backend, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    RETURNS (Promptable): Promptable wrapper for Minichain.
    """

    def init() -> api.Promptable:
        return api.MiniChain(backend, prompt)

    return init


@spacy.registry.llm("spacy.api.LangChain.v1")
def api_langchain(
    backend: str,
    prompt: Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]],
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in langchain.llms, e. g. OpenAI.
    prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    RETURNS (Promptable): Promptable wrapper for LangChain.
    """

    def init() -> api.Promptable:
        return api.LangChain(backend, prompt)

    return init
