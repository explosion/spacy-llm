from typing import Callable, Iterable

from ....registry import registry
from .model import NoOpModel


@registry.llm_models("spacy.NoOp.v1")
def noop() -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns NoOpModel.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): NoOp model instance for test purposes.
    """
    return NoOpModel()
