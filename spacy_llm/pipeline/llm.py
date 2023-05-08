import typing
import warnings
from pathlib import Path

# fmt: off
from typing import Callable, Iterable, Tuple, Iterator, cast, TypeVar, Union, Dict, Optional, Any
# fmt: on

import spacy
from spacy import Language, Vocab
from spacy.pipeline import Pipe
from spacy.tokens import Doc

from .. import registry  # noqa: F401
from ..cache import Cache

_Prompt = TypeVar("_Prompt")
_Response = TypeVar("_Response")
_PromptGenerator = Callable[[Iterable[Doc]], Iterable[_Prompt]]
_PromptExecutor = Callable[[Iterable[_Prompt]], Iterable[_Response]]
_ResponseParser = Callable[[Iterable[Doc], Iterable[_Response]], Iterable[Doc]]
_CacheConfigType = Dict[str, Union[Optional[str], bool, int]]


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
        "cache": {"path": None, "batch_size": 64, "max_n_batches_in_mem": 4},
    },
)
def make_llm(
    nlp: Language,
    name: str,
    task: Tuple[_PromptGenerator, _ResponseParser],
    backend: _PromptExecutor,
    cache: _CacheConfigType,
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    task (Tuple[
        Callable[[Iterable[Doc]], Iterable[_Prompt]],
        Callable[[Iterable[Doc], Iterable[_Response]], Iterable[Doc]]
    ]): Tuple of (1) templating Callable (injecting Doc data into a prompt template and returning one fully specified
        prompt per passed Doc instance) and (2) parsing callable (parsing LLM responses and updating Doc instances with
        the extracted information).
    backend (Callable[[Iterable[_Prompt]], Iterable[_Response]]]): Callable querying the specified LLM API.
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
        template=task[0],
        parse=task[1],
        backend=backend,
        cache=cache,
        vocab=nlp.vocab,
    )


def _validate_types(
    task: Tuple[_PromptGenerator, _ResponseParser], backend: _PromptExecutor
) -> None:
    # Inspect the types of the three main parameters to ensure they match internally
    # Raises an error or prints a warning if something looks wrong/odd.
    type_hints = {
        "template": typing.get_type_hints(task[0]),
        "parse": typing.get_type_hints(task[1]),
        "backend": typing.get_type_hints(backend),
    }

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
            "The 'parse' function should have two input arguments and one return value."
        )
    for k in type_hints["parse"]:
        # find the 'prompt_responses' var without assuming its name
        type_k = type_hints["parse"][k]
        if type_k is not typing.Iterable[spacy.tokens.doc.Doc]:
            parse_input = type_hints["parse"][k]

    template_output = type_hints["template"]["return"]

    # Ensure that the template returns the same type as expected by the backend
    if template_output != backend_input and backend_input != Iterable[Any]:
        warnings.warn(
            f"Type returned from `task[0]` (`{template_output}`) doesn't match type expected by "
            f"`backend` (`{backend_input}`)."
        )

    # Ensure that the parser expects the same type as returned by the backend
    if parse_input != backend_output and backend_output != Iterable[Any]:
        warnings.warn(
            f"Type returned from `backend` (`{backend_output}`) doesn't match type expected by "
            f"`parse` (`{parse_input}`)."
        )


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        vocab: Vocab,
        template: _PromptGenerator,
        parse: _ResponseParser,
        backend: _PromptExecutor,
        cache: _CacheConfigType,
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        vocab (Vocab): Pipeline vocabulary.
        template (Callable[[Iterable[Doc]], Iterable[_Prompt]]): Callable injecting Doc data into a prompt template and
            returning one fully specified prompt per passed Doc instance.
        parse (Callable[[Iterable[Doc], Iterable[_Response]], Iterable[Doc]]): Callable parsing LLM responses and
            updating Doc instances with the extracted information.
        backend (Callable[[Iterable[_Prompt]], Iterable[_Response]]]): Callable querying the specified LLM API.
        cache (Dict[str, Union[Optional[str], bool, int]]): Cache config. If the cache directory `cache["path"]` is
            None, no data will be cached. If a path is set, processed docs will be serialized in the cache directory as
            binary .spacy files. Docs found in the cache directory won't be reprocessed.
        """
        self._name = name
        self._template = template
        self._parse = parse
        self._backend = backend
        self._cache = Cache(
            path=cache["path"],  # type: ignore
            batch_size=int(cache["batch_size"]),  # type: ignore
            max_n_batches_in_mem=int(cache["max_n_batches_in_mem"]),  # type: ignore
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
            self._cache.add(docs[0])

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
                        yield self._cache[doc]
                    else:
                        yield next(modified_docs)
            except Exception as e:
                error_handler(self._name, self, doc_batch, e)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """
        return spacy.util.to_bytes(
            {},
            exclude,
        )

    def from_bytes(self, bytes_data: bytes, *, exclude=tuple()) -> "LLMWrapper":
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """

        spacy.util.from_bytes(
            bytes_data,
            {},
            exclude,
        )

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
