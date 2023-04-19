from pathlib import Path
from typing import Protocol, Iterable, Tuple, cast, Union, Any

from spacy.util import SimpleFrozenList


class Promptable(Protocol):
    """Promptable objects should implement a prompt() method executing multiple prompts and returning the responses."""

    def __call__(self, prompts: Iterable[Any]) -> Iterable[Any]:
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
