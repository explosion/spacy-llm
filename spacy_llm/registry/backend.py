from typing import Callable, Any, Dict, Iterable

import spacy
from spacy.util import SimpleFrozenDict

from ..compat import langchain, minichain


@spacy.registry.llm_queries("spacy.RunMiniChain.v1")
def query_minichain() -> Callable[
    ["minichain.backend.Backend", Iterable[str]], Iterable[str]
]:
    """Returns query Callable for MiniChain.
    RETURNS (Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]): Callable executing simple prompts
        on the specified MiniChain backend.
    """

    def prompt(
        backend: "minichain.backend.Backend", prompts: Iterable[str]
    ) -> Iterable[str]:
        @minichain.prompt(backend)
        def _prompt(mc_backend: "minichain.backend", prompt_text: str) -> str:
            return mc_backend(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt


@spacy.registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> Callable[
    ["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]
]:
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]]:): Callable executing simple prompts on
        the specified LangChain backend.
    """

    def prompt(
        backend: "langchain.llms.BaseLLM", prompts: Iterable[str]
    ) -> Iterable[str]:
        return [backend(pr) for pr in prompts]

    return prompt


@spacy.registry.llm_backends("spacy.MiniChain.v1")
def backend_minichain(
    api: str,
    query: Callable[
        ["minichain.backend.Backend", Iterable[str]], Iterable[str]
    ] = query_minichain(),
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable using MiniChain backend to prompt specified API.
    api (str): Name of any API/class in minichain.backend, e. g. "OpenAI".
    query (Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]): Callable implementing querying this
        API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable querying the specified API using the
        specified backend.
    """

    if hasattr(minichain.backend, api):
        backend = getattr(minichain.backend, api)(**config)

        def _query(prompts: Iterable[str]) -> Iterable[str]:
            return query(backend, prompts)

        return _query
    else:
        raise KeyError(
            f"The requested API {api} is not available in `minichain.backend`."
        )


@spacy.registry.llm_backends("spacy.LangChain.v1")
def backend_langchain(
    api: str,
    query: Callable[
        ["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]
    ] = query_minichain(),
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[Any]], Iterable[Any]]:
    """Returns Callable using MiniChain backend to prompt specified API.
    api (str): Name of any API/class in langchain.llms.type_to_cls_dict, e. g. "openai".
    query (Callable[["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]]): Callable implementing querying this
        API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the langchain.llms.BaseLLM
        instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable using the querying the specified API using the
        specified backend.
    """
    api = api.lower()

    if api in langchain.llms.type_to_cls_dict:
        backend = langchain.llms.type_to_cls_dict[api](**config)

        def _query(prompts: Iterable[Any]) -> Iterable[Any]:
            return query(backend, prompts)

        return _query
    else:
        raise KeyError(
            f"The requested API {api} is not available in `langchain.llms.type_to_cls_dict`."
        )
