from pathlib import Path
from typing import Protocol, Iterable, Tuple, cast, Union, Any

import minichain
import spacy
import srsly
from spacy.util import SimpleFrozenList


class Promptable(Protocol):
    """Promptable objects should implement a prompt() method executing multiple prompts and returning the responses."""

    def prompt(self, prompts: Iterable[Any]) -> Iterable[Any]:
        """Prompt LLM.
        prompts (Iterable[Any]): List of prompts to execute without modifications.
        """

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        """Serialize instance to a bytestring.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized object.
        """

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "Promptable":
        """Load instance from a bytestring.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (Promptable): The loaded object.
        """

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """Serialize instance to disk.
        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        """

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "Promptable":
        """Load instance from disk. Modifies the object in place and returns it.
        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (Promptable): The modified object.
        """


class PromptableMiniChain:
    def __init__(self, backend: str):
        """Initialize wrapper for MiniChain.
        backend (str): Name of any backend class in minichain.backend, e. g. OpenAI.
        """
        self._backend_id = backend
        self._backend: minichain.backend.Backend = getattr(minichain.backend, backend)

    def prompt(self, prompts: Iterable[str]) -> Iterable[str]:
        @minichain.prompt(self._backend())
        def _prompt(model: minichain.backend, prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps({"backend": self._backend_id})

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "PromptableMiniChain":
        self._backend_id = srsly.msgpack_loads(bytes_data)["backend"]
        self._backend = getattr(minichain.backend, self._backend_id)

        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(path, {"backend_id": self._backend_id})

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "PromptableMiniChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        self._backend_id = srsly.read_json(path)["backend_id"]
        self._backend = getattr(minichain.backend, self._backend_id)

        return self
