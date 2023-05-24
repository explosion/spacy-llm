from typing import Any, Callable, Dict, Iterable, Optional, Type
from typing import TYPE_CHECKING

from spacy.util import SimpleFrozenDict

from ..compat import has_langchain, langchain
from ..registry import registry

if TYPE_CHECKING and has_langchain:
    from langchain.llms.base import BaseLLM  # type: ignore[import]


def _check_installation() -> None:
    """Checks whether `langchain` is installed. Raises an error otherwise."""
    if not has_langchain:
        raise ValueError(
            "The LangChain backend requires `langchain` to be installed, which it is not. See "
            "https://github.com/hwchase17/langchain for installation instructions."
        )


@registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> Callable[["BaseLLM", Iterable[Any]], Iterable[Any]]:
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]:): Callable executing simple prompts on
        the specified LangChain backend.
    """
    return _prompt_langchain


def _prompt_langchain(backend: "BaseLLM", prompts: Iterable[Any]) -> Iterable[Any]:
    return [backend(pr) for pr in prompts]


@registry.llm_backends("spacy.LangChain.v1")
def backend_langchain(
    api: str,
    query: Optional[Callable[["BaseLLM", Iterable[str]], Iterable[str]]] = None,
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
    _check_installation()

    # langchain.llms.type_to_cls_dict contains all API names in lowercase, so this allows to be more forgiving and make
    # "OpenAI" work as well "openai".
    api = api.lower()
    type_to_cls_dict: Dict[str, Type[BaseLLM]] = langchain.llms.type_to_cls_dict

    if api in type_to_cls_dict:
        lc_backend = type_to_cls_dict[api](**config)
        query_fn = query_langchain() if query is None else query
        return LangChainBackend(lc_backend, query_fn).query
    else:
        raise KeyError(
            f"The requested API {api} is not available in `langchain.llms.type_to_cls_dict`."
        )


class LangChainBackend:

    def __init__(self, backend: "BaseLLM", query_fn: Callable[["BaseLLM", Iterable[Any]], Iterable[Any]]) -> None:
        self._backend = backend
        self._query_fn = query_fn

    def query(self, prompts: Iterable[Any]) -> Iterable[Any]:
        return self._query_fn(self._backend, prompts)
