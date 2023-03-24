from typing import Callable, Optional, Iterable, Tuple, Literal

import minichain
import srsly
from spacy import Language, registry
from spacy.pipeline import Pipe
from spacy.tokens import Doc


@registry.misc("spacy.DummyPrompt.v1")
def dummy_prompt() -> Callable[[minichain.backend, Doc], str]:
    template = "What is {value} times three? Respond only with the exact number."

    def prompt(model: minichain.backend, doc: Doc) -> str:
        # todo is there a reason to split mapping from prompting here?
        return model(template.format(value=len(doc))).run()

    return prompt


@Language.factory(
    "llm",
    # todo requires and assigns isn't possible to know beforehand. Acceptable to leave empty? Way to set it from config?
    requires=[],
    assigns=[],
    default_config={
        "backend": "openai",
        "response_field": "llm_response",
        "prompt": {"@misc": "spacy.DummyPrompt.v1"},
    },
)
def make_llm(backend: Literal["openai"], response_field: Optional[str]) -> "LLMWrapper":
    """Construct an LLM component.

    backend (str): API backend.
    response_field (Optional[str]): Field in which to store full LLM response. If None, responses are not stored.
    RETURNS (LLMWrapper): LLM instance.
    """
    return LLMWrapper(backend=backend, response_field=response_field)


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self, backend: Literal["openai"], response_field: Optional[str]
    ) -> None:
        """
        Object managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.
        api (Callable[[], API]): Callable generating an API instance.
        response_field (Optional[str]): Field in which to store full LLM response. If None, responses are not stored.
        """
        self._backend = backend
        self._response_field = response_field

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc and set the specified elements.

        doc (Doc): The document to process.
        RETURNS (Doc): The processed Doc.
        """
        return doc

    def predict(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Apply the pipe to a batch of docs, without modifying them.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS (Iterable[Doc]): The predictions for each document.
        """
        return docs

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
