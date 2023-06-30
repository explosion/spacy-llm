from typing import Any, Callable, Dict, Iterable, Optional, Type

from spacy.util import SimpleFrozenDict

from ...compat import has_langchain, langchain
from ...registry import registry


class LangChain:
    def __init__(
        self,
        name: str,
        api: str,
        config: Dict[Any, Any],
        query: Callable[
            ["langchain.llms.base.BaseLLM", Iterable[Any]],
            Iterable[Any],
        ],
    ):
        """Initializes model instance for integration APIs.
        name (str): Name of LangChain model to instantiate.
        api (str): Name of class/API.
        config (Dict[Any, Any]): Config passed on to LangChain model.
        query (Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]]): Callable executing LLM prompts when
            supplied with the `integration` object.
        """
        self._langchain_model = LangChain.get_type_to_cls_dict()[api](
            model_name=name, **config
        )
        self.query = query
        self._check_installation()

    @staticmethod
    def get_type_to_cls_dict() -> Dict[str, Type["langchain.llms.base.BaseLLM"]]:
        """Returns langchain.llms.type_to_cls_dict.
        RETURNS (Dict[str, Type[langchain.llms.base.BaseLLM]]): langchain.llms.type_to_cls_dict.
        """
        return langchain.llms.type_to_cls_dict

    def __call__(self, prompts: Iterable[Any]) -> Iterable[Any]:
        """Executes prompts on specified API.
        prompts (Iterable[Any]): Prompts to execute.
        RETURNS (Iterable[Any]): API responses.
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
    def _langchain_model_maker(class_id: str):
        def langchain_model(
            name: str,
            query: Optional[
                Callable[["langchain.llms.base.BaseLLM", Iterable[str]], Iterable[str]]
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
            except ImportError as err:
                raise ValueError(
                    f"Failed to instantiate LangChain model {langchain_class_id}. Ensure all necessary dependencies "
                    f"are installed."
                ) from err

        return langchain_model

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

        for class_id, cls in LangChain.get_type_to_cls_dict().items():
            registry.llm_models.register(
                f"langchain.{cls.__name__}.v1",
                func=LangChain._langchain_model_maker(class_id=class_id),
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
