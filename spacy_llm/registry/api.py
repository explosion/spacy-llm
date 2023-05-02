from typing import Callable, Any, Dict, Iterable, Tuple

from .util import registry
from .prompt import langchain_simple_prompt, minichain_simple_prompt
from ..compat import langchain, minichain


@registry.apis("spacy-llm.MiniChain.v1")
def api_minichain(
    backend: str,
    config: Dict[Any, Any],
    prompt: Callable[
        ["minichain.backend", Iterable[Any]], Iterable[Any]
    ] = minichain_simple_prompt(),
) -> Tuple[
    Callable[[], "minichain.Backend"],
    Callable[["minichain.backend", Iterable[Any]], Iterable[Any]],
]:
    """Returns promptable minichain.Backend instance.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    prompt (Callable[[minichain.backend, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    RETURNS (Tuple[Callable[[], "minichain.Backend"], Callable[["minichain.backend", Iterable[Any]], Iterable[Any]]]):
        (1) Callable generating a minichain.Backend instance, (2) Callable executing prompts.
    """

    def init() -> "minichain.Backend":
        if hasattr(minichain.backend, backend):
            return getattr(minichain.backend, backend)(**config)
        else:
            raise KeyError(
                f"The requested backend {backend} is not available in `minichain.backend`."
            )

    return init, prompt


@registry.apis("spacy-llm.LangChain.v1")
def api_langchain(
    backend: str,
    config: Dict[Any, Any],
    prompt: Callable[
        ["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]
    ] = langchain_simple_prompt(),
) -> Tuple[
    Callable[[], "langchain.llms.BaseLLM"],
    Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]],
]:
    """Returns promptable langchain.llms.BaseLLM instance.
    backend (str): Name of any backend class in langchain.llms, e. g. OpenAI.
    backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the langchain.llms.BaseLLM
        instance.
    prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
    RETURNS (
        Tuple[
            Callable[[], "langchain.llms.BaseLLM"],
            Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]]
    ): (1) Callable generating a minichain.Backend instance, (2) Callable executing prompts.
    """

    def init() -> langchain.llms.BaseLLM:
        if backend in langchain.llms.type_to_cls_dict:
            return langchain.llms.type_to_cls_dict[backend](**config)
        else:
            raise KeyError(
                f"The requested backend {backend} is not available in `langchain.llms.type_to_cls_dict`."
            )

    return init, prompt
