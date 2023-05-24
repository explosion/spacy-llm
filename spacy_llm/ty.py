from typing import Any, Callable, Iterable, Optional

from spacy import Vocab
from spacy.tokens import Doc

from .compat import Protocol, runtime_checkable


_Prompt = Any
_Response = Any

PromptExecutor = Callable[[Iterable[_Prompt]], Iterable[_Response]]


@runtime_checkable
class LLMTask(Protocol):
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[_Prompt]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[_Response]
    ) -> Iterable[Doc]:
        ...


@runtime_checkable
class Cache(Protocol):
    @property
    def vocab(self) -> Optional[Vocab]:
        """Vocab used for deserializing docs.
        RETURNS (Vocab): Vocab used for deserializing docs.
        """

    @vocab.setter
    def vocab(self, vocab: Vocab) -> None:
        """Set vocab to use for deserializing docs.
        vocab (Vocab): Vocab to use for deserializing docs.
        """

    def add(self, doc: Doc) -> None:
        """Adds processed doc to cache (or to a queue that is added to the cache at a later point)
        doc (Doc): Doc to add to persistence queue.
        """

    def __contains__(self, doc: Doc) -> bool:
        """Checks whether doc has been processed and cached.
        doc (Doc): Doc to check for.
        RETURNS (bool): Whether doc has been processed and cached.
        """

    def __getitem__(self, doc: Doc) -> Optional[Doc]:
        """Loads doc from cache. If doc is not in cache, None is returned.
        doc (Doc): Unprocessed doc whose processed equivalent should be returned.
        RETURNS (Optional[Doc]): Cached and processed version of doc, if available. Otherwise None.
        """
