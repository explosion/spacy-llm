import dataclasses
from typing import Iterable, TypeVar, Callable, Any

# Type of prompts returned from Task.generate_prompts().
_PromptType = TypeVar("_PromptType")
# Type of responses returned from query function.
_ResponseType = TypeVar("_ResponseType")


@dataclasses.dataclass
class Backend:
    integration: Any
    query: Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]]

    def __call__(self, prompts: Iterable[_PromptType]) -> Iterable[_ResponseType]:
        """Executes prompts on specified API.
        prompts (Iterable[_PromptType]): Prompts to execute.
        RETURNS (Iterable[__ResponseType]): API responses.
        """
        return self.query(self.integration, prompts)
