from typing import Any, Callable, Iterable, TypeVar

# Type of prompts returned from Task.generate_prompts().
_PromptType = TypeVar("_PromptType")
# Type of responses returned from query function.
_ResponseType = TypeVar("_ResponseType")


class RemoteBackend:
    def __init__(
        self,
        integration: Any,
        query: Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]],
    ):
        """Initializes Backend instance for remote APIs.
        integration (Any): Object/Callable enabling LLM calls through third-party libraries. This can be a HuggingFace
            model, a LangChain API class, or anything else able to execute LLM prompts directly or indirectly.
        query (Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]]): Callable executing LLM prompts when
            supplied with the `integration` object.
        """
        self._integration = integration
        self.query = query

    def __call__(self, prompts: Iterable[_PromptType]) -> Iterable[_ResponseType]:
        """Executes prompts on specified API.
        prompts (Iterable[_PromptType]): Prompts to execute.
        RETURNS (Iterable[_ResponseType]): API responses.
        """
        return self.query(self._integration, prompts)
