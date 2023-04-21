from pathlib import Path
from typing import Callable, Iterable, Tuple, Iterator, Any, cast

import spacy
from spacy import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc

from .. import registry  # noqa: F401
from ..api import Promptable


@Language.factory(
    "llm",
    requires=[],
    assigns=[],
    default_config={
        "template": {"@llm": "spacy.template.Dummy.v1"},
        "api": {
            "@llm": "spacy.api.MiniChain.v1",
            "backend": "OpenAI",
            "prompt": {"@llm": "spacy.prompt.MiniChainSimple.v1"},
            "backend_config": {},
        },
        "parse": {"@llm": "spacy.parse.Dummy.v1"},
    },
)
def make_llm(
    nlp: Language,
    name: str,
    template: Callable[[Iterable[Doc]], Iterable[Any]],
    api: Callable[[], Promptable],
    parse: Callable[[Iterable[Doc], Iterable[Any]], Iterable[Doc]],
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    template (Callable[[Iterable[Doc]], Iterable[Any]]): Returns Callable injecting Doc data into a prompt template and
        returning one fully specified prompt per passed Doc instance.
    api (Callable[[], Promptable]): Returns Callable generating a Promptable object.
    parse (Callable[[Iterable[Doc], Iterable[Any]], Iterable[Doc]]): Returns Callable parsing LLM
        responses and updating Doc instances with the extracted information.
    RETURNS (LLMWrapper): LLM instance.
    """
    return LLMWrapper(
        name=name,
        template=template,
        api=api,
        parse=parse,
    )


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        template: Callable[[Iterable[Doc]], Iterable[Any]],
        api: Callable[[], Promptable],
        parse: Callable[[Iterable[Doc], Iterable[Any]], Iterable[Doc]],
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        template (Callable[[Iterable[Doc]], Iterable[Any]]): Returns Callable injecting Doc data into a prompt template
            and returning one fully specified prompt per passed Doc instance.
        api (Callable[[], Promptable]): Returns Callable generating a Promptable object.
        parse (Callable[[Iterable[Doc], Iterable[Any]], Iterable[Doc]]): Returns Callable parsing LLM
            responses and updating Doc instances with the extracted information.
        """
        self._name = name
        self._template = template
        self._api = api()
        self._parse = parse

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc instance.

        doc (Doc): The Doc instance to process.
        RETURNS (Doc): The processed Doc.
        """
        docs = list(
            self._parse(
                [doc],
                self._api(self._template([doc])),
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
                    self._api(self._template(doc_batch)),
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
            {
                "api": self._api.to_bytes,
            },
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
            {"api": lambda b: self._api.from_bytes(b)},
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
            {
                "api": lambda p: self._api.to_disk(p),
            },
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
            {"api": lambda p: self._api.from_disk(p)},
            exclude,
        )

        return self
