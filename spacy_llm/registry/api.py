from typing import Callable, Iterable, Any, Dict

import spacy

from .. import api
from ..compat import langchain, minichain


@spacy.registry.llm("spacy.api.MiniChain.v1")
def api_minichain(
    backend: str,
    prompt: Callable[["minichain.Backend", Iterable[Any]], Iterable[Any]],
    backend_config: Dict[Any, Any],
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    prompt (Callable[[minichain.Backend, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    RETURNS (Promptable): Promptable wrapper for Minichain.
    """
    return lambda: api.MiniChain(
        backend=backend, prompt=prompt, backend_config=backend_config
    )


@spacy.registry.llm("spacy.api.LangChain.v1")
def api_langchain(
    backend: str,
    prompt: Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]],
    backend_config: Dict[Any, Any],
) -> Callable[[], api.Promptable]:
    """Returns Promptable wrapper for Minichain.
    backend (str): Name of any backend class in langchain.llms, e. g. OpenAI.
    prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the langchain.llms.BaseLLM
        instance.
    RETURNS (Promptable): Promptable wrapper for LangChain.
    """
    return lambda: api.LangChain(
        backend=backend, prompt=prompt, backend_config=backend_config
    )
