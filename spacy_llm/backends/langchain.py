from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable

from spacy.util import SimpleFrozenDict

from ..compat import has_langchain
from ..registry import registry

if TYPE_CHECKING and has_langchain:
    from langchain.llms import type_to_cls_dict
    from langchain.llms.base import BaseLLM


def _check_installation() -> None:
    """Checks whether `langchain` is installed. Raises an error otherwise."""
    if not has_langchain:
        raise ValueError(
            "The LangChain backend requires `langchain` to be installed, which it is not. See "
            "https://github.com/hwchase17/langchain for installation instructions."
        )


@registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> Callable[["BaseLLM", Iterable[str]], Iterable[str]]:
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]]:): Callable executing simple prompts on
        the specified LangChain backend.
    """
    _check_installation()

    def prompt(backend: "BaseLLM", prompts: Iterable[str]) -> Iterable[str]:
        return [backend(pr) for pr in prompts]

    return prompt


@registry.llm_backends("spacy.LangChain.v1")
def backend_langchain(
    api: str,
    query: Callable[["BaseLLM", Iterable[str]], Iterable[str]] = query_langchain(),
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

    if api in type_to_cls_dict:
        backend = type_to_cls_dict[api](**config)

        def _query(prompts: Iterable[Any]) -> Iterable[Any]:
            return query(backend, prompts)

        return _query
    else:
        raise KeyError(
            f"The requested API {api} is not available in `langchain.llms.type_to_cls_dict`."
        )
