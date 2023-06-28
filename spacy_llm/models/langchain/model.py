import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar

from spacy.util import SimpleFrozenDict

from ...compat import has_langchain, langchain
from ...registry import registry

# Type of prompts returned from Task.generate_prompts().
_PromptType = TypeVar("_PromptType")
# Type of responses returned from query function.
_ResponseType = TypeVar("_ResponseType")


class LangChain:
    TYPE_TO_CLS_DICT: Dict[
        str, Type[langchain.llms.base.BaseLLM]
    ] = langchain.llms.type_to_cls_dict

    def __init__(
        self,
        name: str,
        api: str,
        config: Dict[Any, Any],
        query: Callable[
            ["langchain.llms.base.BaseLLM", Iterable[_PromptType]],
            Iterable[_ResponseType],
        ],
    ):
        """Initializes model instance for integration APIs.
        name (str): Name of LangChain model to instantiate.
        api (str): Name of class/API.
        config (Dict[Any, Any]): Config passed on to LangChain model.
        query (Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]]): Callable executing LLM prompts when
            supplied with the `integration` object.
        """
        self._langchain_model = LangChain.TYPE_TO_CLS_DICT[api](
            model_name=name, **config
        )
        self.query = query
        self._check_installation()

    def __call__(self, prompts: Iterable[_PromptType]) -> Iterable[_ResponseType]:
        """Executes prompts on specified API.
        prompts (Iterable[_PromptType]): Prompts to execute.
        RETURNS (Iterable[_ResponseType]): API responses.
        """
        return self.query(self._langchain_model, prompts)

    @staticmethod
    def query_langchain(
        model: "langchain.llms.base.BaseLLM", prompts: Iterable[Any]
    ) -> Iterable[Any]:
        """Query LangChain model naively.
        model (langchain.llms.base.BaseLLM): LangChain model.
        prompts (Iterable[Any]): Prompts to execute.
        RETURNS (Iterable[Any]): LLM responses.
        """
        return [model(pr) for pr in prompts]

    @staticmethod
    def _check_installation() -> None:
        """Checks whether `langchain` is installed. Raises an error otherwise."""
        if not has_langchain:
            raise ValueError(
                "The LangChain model requires `langchain` to be installed, which it is not. See "
                "https://github.com/hwchase17/langchain for installation instructions."
            )

    @staticmethod
    def register_models() -> None:
        """Registers APIs supported by langchain (one API is registered as one model).
        Doesn't attempt to register anything if LangChain isn't installed or at least one LangChain model has been
        registered already.
        """
        if not has_langchain or any(
            [
                (handle.startswith("langchain"))
                for handle in registry.llm_models.get_all()
            ]
        ):
            return

        for class_id, cls in LangChain.TYPE_TO_CLS_DICT.items():

            def langchain_model(
                name: str,
                query: Optional[
                    Callable[
                        ["langchain.llms.base.BaseLLM", Iterable[str]], Iterable[str]
                    ]
                ] = None,
                config: Dict[Any, Any] = SimpleFrozenDict(),
                langchain_class_id: str = class_id,
            ) -> Optional[Callable[[Iterable[Any]], Iterable[Any]]]:
                try:
                    return LangChain(
                        name=name,
                        api=langchain_class_id,
                        config=config,
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


@registry.llm_queries("spacy.CallLangChain.v1")
def query_langchain() -> (
    Callable[["langchain.llms.base.BaseLLM", Iterable[Any]], Iterable[Any]]
):
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]]:): Callable executing simple prompts on
        the specified LangChain model.
    """
    return LangChain.query_langchain
