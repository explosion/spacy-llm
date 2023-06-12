from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import spacy
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab

from .. import registry  # noqa: F401
from ..compat import TypedDict
from ..ty import Cache, LLMTask, PromptExecutor, validate_types


class CacheConfigType(TypedDict):
    path: Optional[Path]
    batch_size: int
    max_batches_in_mem: int


@Language.factory(
    "llm",
    requires=[],
    assigns=[],
    default_config={
        "task": None,
        "backend": {
            "@llm_backends": "spacy.REST.v1",
            "api": "OpenAI",
            "config": {"model": "gpt-3.5-turbo"},
            "strict": True,
        },
        "cache": {
            "@llm_misc": "spacy.BatchCache.v1",
            "path": None,
            "batch_size": 64,
            "max_batches_in_mem": 4,
        },
    },
)
def make_llm(
    nlp: Language,
    name: str,
    task: Optional[LLMTask],
    backend: PromptExecutor,
    cache: Cache,
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    task (Optional[LLMTask]): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
        structured information and set that back on the docs.
    backend (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
    cache (Cache): Cache to use for caching prompts and responses per doc (batch).
    """
    if task is None:
        raise ValueError(
            "Argument `task` has not been specified, but is required (e. g. {'@llm_tasks': "
            "'spacy.NER.v2'})."
        )
    validate_types(task, backend)

    return LLMWrapper(
        name=name,
        task=task,
        backend=backend,
        cache=cache,
        vocab=nlp.vocab,
    )


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        vocab: Vocab,
        task: LLMTask,
        backend: PromptExecutor,
        cache: Cache,
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        vocab (Vocab): Pipeline vocabulary.
        task (Optional[LLMTask]): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
            structured information and set that back on the docs.
        backend (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
        cache (Cache): Cache to use for caching prompts and responses per doc (batch).
        """
        self._name = name
        self._task = task
        self._backend = backend
        self._cache = cache
        self._cache.vocab = vocab

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc instance.

        doc (Doc): The Doc instance to process.
        RETURNS (Doc): The processed Doc.
        """
        docs = self._process_docs([doc])
        assert isinstance(docs[0], Doc)
        return docs[0]

    def score(
        self, examples: Iterable[Example], **kwargs
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Score a batch of examples.

        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The scores.

        DOCS: https://spacy.io/api/pipe#score
        """
        if hasattr(self._task, "scorer") and self._task.scorer is not None:
            return self._task.scorer(examples)
        return {}

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the LLM prompt to a stream of documents.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        """
        error_handler = self.get_error_handler()
        for doc_batch in spacy.util.minibatch(stream, batch_size):
            try:
                yield from iter(self._process_docs(doc_batch))
            except Exception as e:
                error_handler(self._name, self, doc_batch, e)

    def _process_docs(self, docs: List[Doc]) -> List[Doc]:
        """Process a batch of docs with the configured LLM backend and task.
        If a cache is configured, only sends prompts to backend for docs not found in cache.

        docs (List[Doc]): Input batch of docs
        RETURNS (List[Doc]): Processed batch of docs with task annotations set
        """
        is_cached = [doc in self._cache for doc in docs]
        noncached_doc_batch = [doc for i, doc in enumerate(docs) if not is_cached[i]]
        modified_docs = iter(())
        if noncached_doc_batch:
            prompts = self._task.generate_prompts(noncached_doc_batch)
            responses = self._backend(prompts)
            modified_docs = iter(
                self._task.parse_responses(noncached_doc_batch, responses)
            )

        final_docs = []
        for i, doc in enumerate(docs):
            if is_cached[i]:
                cached_doc = self._cache[doc]
                assert cached_doc is not None
                final_docs.append(cached_doc)
            else:
                doc = next(modified_docs)
                self._cache.add(doc)
                final_docs.append(doc)

        return final_docs

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """
        return b""

    def from_bytes(self, bytes_data: bytes, *, exclude=tuple()) -> "LLMWrapper":
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        return self

    def to_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> None:
        """Serialize the LLMWrapper to disk.
        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """
        return None

    def from_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LLMWrapper":
        """Load the LLMWrapper from disk.
        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        return self
