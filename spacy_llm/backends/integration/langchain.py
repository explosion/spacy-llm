from typing import Any, Callable, Dict, Iterable, Optional, Type

from spacy.util import SimpleFrozenDict

from ...compat import has_langchain, langchain
from ...registry import registry
from . import Backend


def _check_installation() -> None:
    """Checks whether `langchain` is installed. Raises an error otherwise."""
    if not has_langchain:
        raise ValueError(
            "The LangChain backend requires `langchain` to be installed, which it is not. See "
            "https://github.com/hwchase17/langchain for installation instructions."
        )


@registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> Callable[
    ["langchain.llms.base.BaseLLM", Iterable[Any]], Iterable[Any]
]:
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]:): Callable executing simple prompts on
        the specified LangChain backend.
    """
    return _prompt_langchain


def _prompt_langchain(
    backend: "langchain.llms.base.BaseLLM", prompts: Iterable[Any]
) -> Iterable[Any]:
    return [backend(pr) for pr in prompts]


@registry.llm_backends("spacy.LangChain.v1")
def backend_langchain(
    api: str,
    query: Optional[
        Callable[["langchain.llms.base.BaseLLM", Iterable[str]], Iterable[str]]
    ] = None,
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

    if "model" not in config:
        raise ValueError("The LLM model must be specified in the config.")

    # langchain.llms.type_to_cls_dict contains all API names in lowercase, so this allows to be more forgiving and make
    # "OpenAI" work as well "openai".
    api = api.lower()
    type_to_cls_dict: Dict[
        str, Type[langchain.llms.base.BaseLLM]
    ] = langchain.llms.type_to_cls_dict

    if api in type_to_cls_dict:
        model = config.pop("model")
        return Backend(
            integration=type_to_cls_dict[api](model=model, **config),
            query=query_langchain() if query is None else query,
        )
    else:
        raise KeyError(
            f"The requested API {api} is not available in `langchain.llms.type_to_cls_dict`."
        )
