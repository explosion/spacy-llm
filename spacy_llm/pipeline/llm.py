from pathlib import Path
from typing import Callable, Optional, Iterable, Tuple, Iterator, List, Any, Dict, cast

import minichain
import spacy
import srsly
from spacy import Language, registry
from spacy.pipeline import Pipe
from spacy.tokens import Doc


@registry.misc("spacy.DummyPrompt.v1")
def dummy_prompt() -> Callable[[minichain.backend.Backend, Optional[str], Doc], Doc]:
    """Returns prompt function accepting specified API and document, prompting LLM API, mapping response to the doc
    instance and this instance.
    RETURNS (Callable[[minichain.backend.Backend, Optional[str], Doc], Doc]): Prompt function.
    """
    template = "What is {value} times three? Respond with the exact number ."

    def prompt(
        backend: minichain.backend.Backend, response_field: Optional[str], doc: Doc
    ) -> Doc:
        @minichain.prompt(backend())
        def _prompt(model: minichain.backend) -> str:
            return model(template.format(value=len(doc)))

        response = _prompt().run()
        if response_field:
            setattr(doc._, response_field, response)

        return doc

    return prompt


@registry.misc("spacy.DummyBatchPrompt.v1")
def dummy_batch_prompt() -> Callable[
    [minichain.backend.Backend, Optional[str], Iterable[Doc]], Iterable[Doc]
]:
    """Returns prompt function accepting specified API and documents, prompting LLM API, mapping response to the doc
    instances and returning those instances.
    This particular dummy implementation loops over individual prompts, but real implementations may use particular API
    batching functionality for better performance.
    RETURNS (Callable[[minichain.backend.Backend, Optional[str], Iterable[Doc]], Iterable[Doc]]): Prompt function.
    """
    template = "What is {value} times three? Respond with the exact number."

    def prompt(
        backend: minichain.backend.Backend,
        response_field: Optional[str],
        docs: Iterable[Doc],
    ) -> Iterable[Doc]:
        @minichain.prompt(backend())
        def _prompt(model: minichain.backend, doc: Doc) -> str:
            return model(template.format(value=len(doc)))

        for doc in docs:
            response = _prompt(doc).run()
            if response_field:
                setattr(doc._, response_field, response)

        return docs

    return prompt


@Language.factory(
    "llm",
    # todo requires and assigns isn't possible to know beforehand. Acceptable to leave empty? Way to set it from config?
    requires=[],
    assigns=[],
    default_config={
        "backend": "OpenAI",
        "response_field": "llm_response",
        "prompt": {"@misc": "spacy.DummyPrompt.v1"},
        "batch_prompt": {"@misc": "spacy.DummyBatchPrompt.v1"},
    },
)
def make_llm(
    nlp: Language,
    name: str,
    backend: str,
    response_field: Optional[str],
    prompt: Callable[[minichain.backend.Backend, Optional[str], Doc], Doc],
    batch_prompt: Callable[
        [minichain.backend.Backend, Optional[str], Iterable[Doc]], Iterable[Doc]
    ],
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
    response_field (Optional[str]): Field in which to store full LLM response. If None, responses are not stored.
    prompt (Callable[[minichain.backend.Backend, Optional[str], Doc], Doc]): Callable generating prompt function
        modifying the specified doc.
    batch_prompt (Callable[[minichain.backend, Optional[str], Iterable[Doc]], Iterable[Doc]]): Callable generating
        prompt function modifying the specified docs.
    RETURNS (LLMWrapper): LLM instance.
    """
    return LLMWrapper(
        name=name,
        backend=backend,
        response_field=response_field,
        prompt=prompt,
        batch_prompt=batch_prompt,
    )


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        backend: str,
        response_field: Optional[str],
        prompt: Callable[[minichain.backend.Backend, Optional[str], Doc], Doc],
        batch_prompt: Callable[
            [minichain.backend.Backend, Optional[str], Iterable[Doc]], Iterable[Doc]
        ],
    ) -> None:
        """
        Object managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
        response_field (Optional[str]): Field in which to store full LLM response. If None, responses are not stored.
        prompt (Callable[[Doc, Optional[str]], Doc]): Callable generating prompt function modifying the specified
            doc.
        batch_prompt (Callable[[minichain.backend, Optional[str], Iterable[Doc]], Iterable[Doc]]): Callable generating
            prompt function modifying the specified docs.
        """
        self._name = name
        self._backend_id = backend
        self._backend: minichain.backend.Backend = getattr(
            minichain.backend, self._backend_id
        )
        self._response_field = response_field
        self._prompt = prompt
        self._batch_prompt = batch_prompt

        if not Doc.has_extension(self._response_field):
            Doc.set_extension(self._response_field, default=None)

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc and set the specified elements.

        doc (Doc): The document to process.
        RETURNS (Doc): The processed Doc.
        """
        return self._prompt(self._backend, self._response_field, doc)

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        error_handler (Callable[[str, List[Doc], Exception], Any]): Function that
            deals with a failing batch of documents. The default function just reraises
            the exception.
        YIELDS (Doc): Processed documents in order.

        DOCS: https://spacy.io/api/pipe#pipe
        """
        doc_batch: List[Doc] = []
        for doc in stream:
            doc_batch.append(doc)
            if len(doc_batch) % batch_size == 0:
                for modified_doc in self._batch_prompt(
                    self._backend, self._response_field, doc_batch
                ):
                    yield modified_doc
                doc_batch = []

        # Run prompt for last, incomplete batch.
        for modified_doc in self._batch_prompt(
            self._backend, self._response_field, doc_batch
        ):
            yield modified_doc

    def _to_serializable_dict(self, exclude: Tuple[str]) -> Dict[str, Any]:
        """Returns dict with serializable properties.
        exclude (Tuple[str]): Names of properties to exclude from serialization.
        RETURNS (Dict[str, Any]): Dict with serializable properties.
        """
        return {
            k: v
            for k, v in {
                "backend_id": self._backend_id,
                "response_field": self._response_field,
            }.items()
            if k not in exclude
        }

    def _from_deserialized_dict(self, cfg: Dict[str, Any], exclude: Tuple[str]) -> None:
        """Set instance value from config dict.
        cfg (Dict[str, Any]): Config dict.
        exclude (Tuple[str]): Names of properties to exclude from deserialization.
        """
        if "backend_id" not in exclude:
            self._backend_id = cfg["backend_id"]
        if "backend" not in exclude:
            self._backend = getattr(minichain.backend, self._backend_id)
        if "response_field" not in exclude:
            self._response_field = cfg["response_field"]

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """

        return srsly.msgpack_dumps(self._to_serializable_dict(exclude))

    def from_bytes(self, bytes_data: bytes, *, exclude=tuple()) -> "LLMWrapper":
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        self._from_deserialized_dict(srsly.msgpack_loads(bytes_data), exclude)

        return self

    def to_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> None:
        """Serialize the LLMWrapper to disk.
        path (Path): A path to a JSON file, which will be created if it doesn’t exist. Paths may be either strings or
            Path-like objects.
        exclude (Tuple): Names of properties to exclude from serialization.
        """
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(path, self._to_serializable_dict(exclude))

    def from_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LLMWrapper":
        """Load the LLMWrapper from disk.
        path (Path): A path to a JSON file. Paths may be either strings or Path-like objects.
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """
        path = spacy.util.ensure_path(path).with_suffix(".json")
        self._from_deserialized_dict(srsly.read_json(path), exclude)

        return self
