import typing
import warnings
from pathlib import Path
from typing import Iterable, Tuple, Iterator, Type
from typing import cast, Union, Dict, Optional, Any

import spacy
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.vocab import Vocab

from .. import registry  # noqa: F401
from ..cache import Cache
from ..ty import LLMTask, PromptExecutor


CacheConfigType = Dict[str, Union[Optional[str], bool, int]]


@Language.factory(
    "llm",
    requires=[],
    assigns=[],
    default_config={
        "task": None,
        "backend": {
            "@llm_backends": "spacy.REST.v1",
            "api": "OpenAI",
            "config": {"model": "text-davinci-003"},
            "strict": True,
        },
        "cache": {"path": None, "batch_size": 64, "max_batches_in_mem": 4},
    },
)
def make_llm(
    nlp: Language,
    name: str,
    task: Optional[LLMTask],
    backend: PromptExecutor,
    cache: CacheConfigType,
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    task (Optional[LLMTask]): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
        structured information and set that back on the docs.
    backend (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
    cache (Dict[str, Union[Optional[str], bool, int]]): Cache config. If the cache directory `cache["path"]` is None, no
        data will be cached. If a path is set, processed docs will be serialized in the cache directory as binary .spacy
        files. Docs found in the cache directory won't be reprocessed.
    """
    if task is None:
        raise ValueError(
            "Argument `task` has not been specified, but is required (e. g. {'@llm_tasks': "
            "'spacy.NER.v1'})."
        )
    _validate_types(task, backend)

    return LLMWrapper(
        name=name,
        task=task,
        backend=backend,
        cache=cache,
        vocab=nlp.vocab,
    )


def _validate_types(task: LLMTask, backend: PromptExecutor) -> None:
    # Inspect the types of the three main parameters to ensure they match internally
    # Raises an error or prints a warning if something looks wrong/odd.
    if not isinstance(task, LLMTask):
        raise ValueError(
            f"A task needs to be of type 'LLMTask' but found {type(task)} instead"
        )
    if not hasattr(task, "generate_prompts"):
        raise ValueError(
            "A task needs to have the following method: generate_prompts(self, docs: Iterable[Doc]) -> Iterable[Any]"
        )
    if not hasattr(task, "parse_responses"):
        raise ValueError(
            "A task needs to have the following method: "
            "parse_responses(self, docs: Iterable[Doc], responses: Iterable[Any]) -> Iterable[Doc]"
        )

    type_hints = {
        "template": typing.get_type_hints(task.generate_prompts),
        "parse": typing.get_type_hints(task.parse_responses),
        "backend": typing.get_type_hints(backend),
    }

    parse_input: Optional[Type] = None
    backend_input: Optional[Type] = None
    backend_output: Optional[Type] = None

    # Validate the 'backend' object
    if not (len(type_hints["backend"]) == 2 and "return" in type_hints["backend"]):
        raise ValueError(
            "The 'backend' function should have one input argument and one return value."
        )
    for k in type_hints["backend"]:
        if k == "return":
            backend_output = type_hints["backend"][k]
        else:
            backend_input = type_hints["backend"][k]

    # validate the 'parse' object
    if not (len(type_hints["parse"]) == 3 and "return" in type_hints["parse"]):
        raise ValueError(
            "The 'task.parse_responses()' function should have two input arguments and one return value."
        )
    for k in type_hints["parse"]:
        # find the 'prompt_responses' var without assuming its name
        type_k = type_hints["parse"][k]
        if type_k is not typing.Iterable[Doc]:
            parse_input = type_hints["parse"][k]

    template_output = type_hints["template"]["return"]

    # Ensure that the template returns the same type as expected by the backend
    if template_output != backend_input and backend_input != Iterable[Any]:
        warnings.warn(
            f"Type returned from `task.generate_prompts()` (`{template_output}`) doesn't match type expected by "
            f"`backend` (`{backend_input}`)."
        )

    # Ensure that the parser expects the same type as returned by the backend
    if parse_input != backend_output and backend_output != Iterable[Any]:
        warnings.warn(
            f"Type returned from `backend` (`{backend_output}`) doesn't match type expected by "
            f"`task.parse_responses()` (`{parse_input}`)."
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
        cache: CacheConfigType,
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        vocab (Vocab): Pipeline vocabulary.
        task (Optional[LLMTask]): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
            structured information and set that back on the docs.
        backend (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
        cache (Dict[str, Union[Optional[str], bool, int]]): Cache config. If the cache directory `cache["path"]` is
            None, no data will be cached. If a path is set, processed docs will be serialized in the cache directory as
            binary .spacy files. Docs found in the cache directory won't be reprocessed.
        """
        self._name = name
        self._template = task.generate_prompts
        self._parse = task.parse_responses
        self._backend = backend
        self._cache = Cache(
            path=cache["path"],  # type: ignore
            batch_size=int(cache["batch_size"]),  # type: ignore
            max_batches_in_mem=int(cache["max_batches_in_mem"]),  # type: ignore
            vocab=vocab,
        )

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc instance.

        doc (Doc): The Doc instance to process.
        RETURNS (Doc): The processed Doc.
        """
        docs = [self._cache[doc]]
        if docs[0] is None:
            docs = list(
                self._parse(
                    [doc],
                    self._backend(self._template([doc])),
                )
            )
            assert len(docs) == 1
            assert isinstance(docs[0], Doc)
            self._cache.add(docs[0])

        assert isinstance(docs[0], Doc)
        return docs[0]

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the LLM prompt to a stream of documents.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        """
        error_handler = self.get_error_handler()
        for doc_batch in spacy.util.minibatch(stream, batch_size):
            is_cached = [doc in self._cache for doc in doc_batch]
            noncached_doc_batch = [
                doc for i, doc in enumerate(doc_batch) if not is_cached[i]
            ]

            try:
                modified_docs = iter(
                    self._parse(
                        doc_batch, self._backend(self._template(noncached_doc_batch))
                    )
                )
                for i, doc in enumerate(doc_batch):
                    if is_cached[i]:
                        doc = self._cache[doc]
                        assert isinstance(doc, Doc)
                        yield doc
                    else:
                        doc = next(modified_docs)
                        self._cache.add(doc)
                        yield doc
            except Exception as e:
                error_handler(self._name, self, doc_batch, e)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """
        return spacy.util.to_bytes({}, exclude)

    def from_bytes(self, bytes_data: bytes, *, exclude=tuple()) -> "LLMWrapper":
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        spacy.util.from_bytes(bytes_data, {}, exclude)
        return self

    def to_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> None:
        """Serialize the LLMWrapper to disk.
        path (Path): A path to a JSON file, which will be created if it doesn’t exist. Paths may be either strings or
            Path-like objects.
        exclude (Tuple): Names of properties to exclude from serialization.
        """
        spacy.util.to_disk(
            spacy.util.ensure_path(path).with_suffix(".json"),
            {},
            exclude,
        )

    def from_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LLMWrapper":
        """Load the LLMWrapper from disk.
        path (Path): A path to a JSON file. Paths may be either strings or Path-like objects.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        spacy.util.from_disk(
            spacy.util.ensure_path(path).with_suffix(".json"),
            {},
            exclude,
        )
        return self
