import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Type

from spacy.util import SimpleFrozenDict

from ...compat import has_langchain, langchain
from ...registry import registry
from . import Remote


def _check_installation() -> None:
    """Checks whether `langchain` is installed. Raises an error otherwise."""
    if not has_langchain:
        raise ValueError(
            "The LangChain model requires `langchain` to be installed, which it is not. See "
            "https://github.com/hwchase17/langchain for installation instructions."
        )


@registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> (
    Callable[["langchain.llms.base.BaseLLM", Iterable[Any]], Iterable[Any]]
):
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]:): Callable executing simple prompts on
        the specified LangChain model.
    """
    return _prompt_langchain


def _prompt_langchain(
    model: "langchain.llms.base.BaseLLM", prompts: Iterable[Any]
) -> Iterable[Any]:
    return [model(pr) for pr in prompts]


def register_langchain_models() -> None:
    """Registers APIs supported by langchain (one API is registered as one model)."""
    type_to_cls_dict: Dict[
        str, Type[langchain.llms.base.BaseLLM]
    ] = langchain.llms.type_to_cls_dict

    for class_id, cls in type_to_cls_dict.items():

        def langchain_model(
            query: Optional[
                Callable[["langchain.llms.base.BaseLLM", Iterable[str]], Iterable[str]]
            ] = None,
            config: Dict[Any, Any] = SimpleFrozenDict(),
            langchain_class_id: str = class_id,
        ) -> Optional[Callable[[Iterable[Any]], Iterable[Any]]]:
            model = config.pop("model")
            try:
                return Remote(
                    integration=type_to_cls_dict[langchain_class_id](
                        model_name=model, **config
                    ),
                    query=query_langchain() if query is None else query,
                )
            except ImportError:
                warnings.warn(
                    f"Failed to instantiate LangChain class {cls.__name__}. Ensure all necessary dependencies are "
                    f"installed."
                )
                return None

        registry.llm_models.register(
            f"langchain.{cls.__name__}.v1", func=langchain_model
        )


# Dynamically register LangChain API classes as individual models, if this hasn't been done yet.
if has_langchain and not any(
    [("langchain" in handle) for handle in registry.llm_models.get_all()]
):
    register_langchain_models()
