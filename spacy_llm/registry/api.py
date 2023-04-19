from typing import Callable, Iterable, Any, Dict

import langchain.llms
import minichain
import spacy

from .. import api


@spacy.registry.llm("spacy.api.MiniChain.v1")
def api_minichain(
    backend: str,
    prompt: Callable[[minichain.Backend, Iterable[Any]], Iterable[Any]],
    core_config: Dict[Any, Any],
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    prompt (Callable[[minichain.Backend, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    core_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    RETURNS (Promptable): Promptable wrapper for Minichain.
    """

    def init() -> api.Promptable:
        return api.MiniChain(backend=backend, prompt=prompt, core_config=core_config)

    return init


@spacy.registry.llm("spacy.api.LangChain.v1")
def api_langchain(
    backend: str,
    prompt: Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]],
    core_config: Dict[Any, Any],
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in langchain.llms, e. g. OpenAI.
    prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    core_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the langchain.llms.BaseLLM
        instance.
    RETURNS (Promptable): Promptable wrapper for LangChain.
    """

    def init() -> api.Promptable:
        return api.LangChain(backend=backend, prompt=prompt, core_config=core_config)

    return init
