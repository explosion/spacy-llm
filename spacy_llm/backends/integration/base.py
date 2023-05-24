import dataclasses
from typing import Iterable, Generic, TypeVar, Callable

# Object/Callable from third-party lib used to execute prompts.
_Integration = TypeVar("_Integration")
# Type of prompts returned from Task.generate_prompts().
_PromptType = TypeVar("_PromptType")
# Type of responses returned from query function.
_ResponseType = TypeVar("_ResponseType")


@dataclasses.dataclass
class Backend(Generic[_Integration, _PromptType, _ResponseType]):
    integration: _Integration
    query: Callable[[_Integration, Iterable[_PromptType]], Iterable[_ResponseType]]

    def __call__(self, prompts: Iterable[_PromptType]) -> Iterable[_ResponseType]:
        """Executes prompts on specified API.
        prompts (Iterable[_PromptType]): Prompts to execute.
        RETURNS (Iterable[__ResponseType]): API responses.
        """
        return self.query(self.integration, prompts)
