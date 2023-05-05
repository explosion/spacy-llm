import typing
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, Tuple, TypeVar, cast

import spacy
from spacy import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc

_Prompt = TypeVar("_Prompt")
_Response = TypeVar("_Response")
_PromptGenerator = Callable[[Iterable[Doc]], Iterable[_Prompt]]
_PromptExecutor = Callable[[Iterable[_Prompt]], Iterable[_Response]]
_ResponseParser = Callable[[Iterable[Doc], Iterable[_Response]], Iterable[Doc]]


@Language.factory(
    "llm",
    requires=[],
    assigns=[],
    default_config={
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "backend": {
            "@llm_backends": "spacy.MiniChain.v1",
            "api": "OpenAI",
            "config": {},
            "query": {"@llm_queries": "spacy.RunMiniChain.v1"},
        },
    },
)
def make_llm(
    nlp: Language,
    name: str,
    task: Tuple[_PromptGenerator, _ResponseParser],
    backend: _PromptExecutor,
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
    """
    # Warn if types don't match.
    type_hints = {
        "template": typing.get_type_hints(task[0]),
        "parse": typing.get_type_hints(task[1]),
        "backend": typing.get_type_hints(backend),
    }
    for source, dest in (
        ("template", ("backend", "prompts")),
        ("backend", ("parse", "prompt_responses")),
    ):
        if type_hints[source]["return"] != type_hints[dest[0]][dest[1]]:
            warnings.warn(
                f"Type returned from `{source}()` (`{type_hints[source]['return']}`) doesn't match type expected by "
                f"`{dest[0]}()` (`{type_hints[dest[0]][dest[1]]}`)."
            )

    return LLMWrapper(name=name, template=task[0], parse=task[1], backend=backend)


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        template: _PromptGenerator,
        parse: _ResponseParser,
        backend: _PromptExecutor,
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        template (Callable[[Iterable[Doc]], Iterable[_Prompt]]): Callable injecting Doc data into a prompt template and
            returning one fully specified prompt per passed Doc instance.
        parse (Callable[[Iterable[Doc], Iterable[_Response]], Iterable[Doc]]): Callable parsing LLM responses and
            updating Doc instances with the extracted information.
        backend (Callable[[Iterable[_Prompt]], Iterable[_Response]]]): Callable querying the specified LLM API.
        """
        self._name = name
        self._template = template
        self._parse = parse
        self._backend = backend

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc instance.

        doc (Doc): The Doc instance to process.
        RETURNS (Doc): The processed Doc.
        """
        docs = list(
            self._parse(
                [doc],
                self._backend(self._template([doc])),
            )
        )
        assert len(docs) == 1
        return docs[0]

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the LLM prompt to a stream of documents.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        """
        error_handler = self.get_error_handler()
        for doc_batch in spacy.util.minibatch(stream, batch_size):
            try:
                for modified_doc in self._parse(
                    doc_batch,
                    self._backend(self._template(doc_batch)),
                ):
                    yield modified_doc
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
