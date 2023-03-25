from typing import Callable, Optional, Iterable, Tuple, Iterator, List

import minichain
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
    template = "What is {value} times three? Respond with the exact number."

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
    This particular dummy implementation just loops over individual prompts, but real ones could concatenate/chain
    prompts for performance improvements.
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
        nlp=nlp,
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
        nlp: Language,
        name: str,
        backend: str,
        response_field: Optional[str],
        prompt: Callable[[minichain.backend.Backend, Optional[str], Doc], Doc],
        batch_prompt: Callable[
            [minichain.backend.Backend, Optional[str], Iterable[Doc]], Iterable[Doc]
        ],
    ) -> None:
        """
        Object managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        nlp (Language): Pipeline.
        name (str): The component instance name, used to add entries to the
            losses during training.
        backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
        response_field (Optional[str]): Field in which to store full LLM response. If None, responses are not stored.
        prompt (Callable[[Doc, Optional[str]], Doc]): Callable generating prompt function modifying the specified
            doc.
        batch_prompt (Callable[[minichain.backend, Optional[str], Iterable[Doc]], Iterable[Doc]]): Callable generating
            prompt function modifying the specified docs.
        """
        self._nlp = nlp
        self._name = name
        self._backend: minichain.backend.Backend = getattr(minichain.backend, backend)
        self._response_field = response_field
        self._prompt = prompt
        self._batch_prompt = batch_prompt

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

    def to_bytes(self, *, exclude: Tuple = tuple()) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        RETURNS (bytes): The serialized object.
        todo
        """
        return srsly.msgpack_dumps({})

    def from_bytes(self, bytes_data: bytes, *, exclude=tuple()):
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        returns (Sentencizer): The loaded object.
        todo
        """
        # cfg = srsly.msgpack_loads(bytes_data)
        # self.punct_chars = set(cfg.get("punct_chars", self.default_punct_chars))
        # self.overwrite = cfg.get("overwrite", self.overwrite)
        return self

    def to_disk(self, path, *, exclude=tuple()):
        """Serialize the LLMWrapper to disk.
        todo
        """
        # path = util.ensure_path(path)
        # path = path.with_suffix(".json")
        # srsly.write_json(path, {"punct_chars": list(self.punct_chars), "overwrite": self.overwrite})

    def from_disk(self, path, *, exclude=tuple()):
        """Load the LLMWrapper from disk.
        todo
        """
        # path = util.ensure_path(path)
        # path = path.with_suffix(".json")
        # cfg = srsly.read_json(path)
        # self.punct_chars = set(cfg.get("punct_chars", self.default_punct_chars))
        # self.overwrite = cfg.get("overwrite", self.overwrite)
        return self
