from typing import Any, Callable, Dict, Iterable, Optional, Type

from confection import SimpleFrozenDict

from ...compat import ExtraError, ValidationError, has_langchain, langchain_community
from ...registry import registry

try:
    from langchain_community import llms  # noqa: F401
except (ImportError, AttributeError):
    llms = None


class LangChain:
    def __init__(
        self,
        name: str,
        api: str,
        config: Dict[Any, Any],
        query: Callable[
            ["langchain_community.llms.BaseLLM", Iterable[Iterable[Any]]],
            Iterable[Iterable[Any]],
        ],
        context_length: Optional[int],
    ):
        """Initializes model instance for integration APIs.
        name (str): Name of LangChain model to instantiate.
        api (str): Name of class/API.
        config (Dict[Any, Any]): Config passed on to LangChain model.
        query (Callable[[langchain_community.llms.BaseLLM, Iterable[Iterable[Any]]], Iterable[Iterable[Any]]]): Callable
            executing LLM prompts when supplied with the model instance.
        context_length (Optional[int]): Context length for this model. Only necessary for sharding. If no context
            length provided, prompts can't be sharded.
        """
        self._langchain_model = LangChain._init_langchain_model(name, api, config)
        self.query = query
        self._context_length = context_length
        self._check_installation()

    @classmethod
    def _init_langchain_model(
        cls, name: str, api: str, config: Dict[Any, Any]
    ) -> "langchain_community.llms.BaseLLM":
        """Initializes langchain model. langchain expects a range of different model ID argument names, depending on the
        model class. There doesn't seem to be a clean way to determine those from the outset, we'll fail our way through
        them.
        Includes error checks for model ID arguments.
        name (str): Name of LangChain model to instantiate.
        api (str): Name of class/API.
        config (Dict[Any, Any]): Config passed on to LangChain model.
        """
        model_init_args = ["model", "model_name", "model_id"]
        for model_init_arg in model_init_args:
            try:
                return cls.get_type_to_cls_dict()[api](
                    **{model_init_arg: name}, **config
                )
            except ValidationError as err:
                if model_init_arg == model_init_args[-1]:
                    # If init error indicates that model ID arg is extraneous: raise error with hint on how to proceed.
                    if any(
                        [
                            rerr
                            for rerr in err.raw_errors
                            if isinstance(rerr.exc, ExtraError)
                            and model_init_arg in rerr.loc_tuple()
                        ]
                    ):
                        raise ValueError(
                            "Couldn't initialize LangChain model with known model ID arguments. Please report this to "
                            "https://github.com/explosion/spacy-llm/issues. Thank you!"
                        ) from err
                    # Otherwise: raise error as-is.
                    raise err

    @staticmethod
    def get_type_to_cls_dict() -> Dict[str, Type["langchain_community.llms.BaseLLM"]]:
        """Returns langchain_community.llms.type_to_cls_dict.
        RETURNS (Dict[str, Type[langchain_community.llms.BaseLLM]]): langchain_community.llms.type_to_cls_dict.
        """
        return {
            llm_id: getattr(langchain_community.llms, llm_id)
            for llm_id in langchain_community.llms.__all__
        }

    def __call__(self, prompts: Iterable[Iterable[Any]]) -> Iterable[Iterable[Any]]:
        """Executes prompts on specified API.
        prompts (Iterable[Iterable[Any]]): Prompts to execute.
        RETURNS (Iterable[Iterable[Any]]): API responses.
        """
        return self.query(self._langchain_model, prompts)

    @staticmethod
    def query_langchain(
        model: "langchain_community.llms.BaseLLM", prompts: Iterable[Iterable[Any]]
    ) -> Iterable[Iterable[Any]]:
        """Query LangChain model naively.
        model (langchain_community.llms.BaseLLM): LangChain model.
        prompts (Iterable[Iterable[Any]]): Prompts to execute.
        RETURNS (Iterable[Iterable[Any]]): LLM responses.
        """
        return [
            [model.invoke(pr) for pr in prompts_for_doc] for prompts_for_doc in prompts
        ]

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
                Callable[
                    ["langchain_community.llms.BaseLLM", Iterable[Iterable[str]]],
                    Iterable[Iterable[str]],
                ]
            ] = None,
            config: Dict[Any, Any] = SimpleFrozenDict(),
            context_length: Optional[int] = None,
            langchain_class_id: str = class_id,
        ) -> Optional[Callable[[Iterable[Iterable[Any]]], Iterable[Iterable[Any]]]]:
            try:
                return LangChain(
                    name=name,
                    api=langchain_class_id,
                    config=config,
                    query=query_langchain() if query is None else query,
                    context_length=context_length,
                )
            except ImportError as err:
                raise ValueError(
                    f"Failed to instantiate LangChain model {langchain_class_id}. Ensure all necessary dependencies "
                    f"are installed."
                ) from err

        return langchain_model

    @property
    def context_length(self) -> Optional[int]:
        """Returns context length in number of tokens for this model.
        RETURNS (Optional[int]): Max. number of tokens in allowed in prompt for the current model. None if unknown.
        """
        return self._context_length

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
    Callable[
        ["langchain_community.llms.BaseLLM", Iterable[Iterable[Any]]],
        Iterable[Iterable[Any]],
    ]
):
    """Returns query Callable for LangChain.
    RETURNS (Callable[["langchain_community.llms.BaseLLM", Iterable[Iterable[Any]]], Iterable[Iterable[Any]]]): Callable
        executing simple prompts on the specified LangChain model.
    """
    return LangChain.query_langchain
