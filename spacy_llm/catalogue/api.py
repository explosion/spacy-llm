from typing import Callable, Any, Dict

from ..util import registry
from ..compat import langchain, minichain


@registry.apis("spacy-llm.MiniChain.v1")
def api_minichain(
    backend: str,
    config: Dict[Any, Any],
) -> Callable[[], "minichain.Backend"]:
    """Returns promptable minichain.Backend instance.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    RETURNS (Callable[[], "minichain.Backend"]): Callable generating a minichain.Backend instance.
    """

    def init() -> "minichain.Backend":
        if hasattr(minichain.backend, backend):
            return getattr(minichain.backend, backend)(**config)
        else:
            raise KeyError(
                f"The requested backend {backend} is not available in `minichain.backend`."
            )

    return init


@registry.apis("spacy-llm.LangChain.v1")
def api_langchain(
    backend: str,
    config: Dict[Any, Any],
) -> Callable[[], "langchain.llms.BaseLLM"]:
    """Returns promptable langchain.llms.BaseLLM instance.
    backend (str): Name of any backend class in langchain.llms, e. g. OpenAI.
    backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the langchain.llms.BaseLLM
        instance.
    RETURNS (Callable[[], "minichain.Backend"]): Callable generating a langchain.llms.BaseLLM instance.
    """

    def init() -> langchain.llms.BaseLLM:
        if backend in langchain.llms.type_to_cls_dict:
            return langchain.llms.type_to_cls_dict[backend](**config)
        else:
            raise KeyError(
                f"The requested backend {backend} is not available in `langchain.llms.type_to_cls_dict`."
            )

    return init
