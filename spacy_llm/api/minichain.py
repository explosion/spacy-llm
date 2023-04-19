from pathlib import Path
from typing import Iterable, Tuple, cast, Union, Callable, Any, Dict

import spacy
import srsly
from spacy.util import SimpleFrozenList

from ..compat import minichain


class MiniChain:
    def __init__(
        self,
        backend: str,
        prompt: Callable[[minichain.Backend, Iterable[str]], Iterable[str]],
        backend_config: Dict[Any, Any],
    ):
        """Initialize wrapper for MiniChain.
        backend (str): Name of any backend class in minichain.backend, e. g. "OpenAI".
        prompt (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]): Callable executing prompts.
        backend_config (Dict[Any, Any]): Not used.
        """
        self._backend_id = backend
        self._backend: minichain.backend.Backend = getattr(minichain.backend, backend)
        self._prompt = prompt

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        return self._prompt(self._backend, prompts)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps({"backend": self._backend_id})

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "MiniChain":
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
    ) -> "MiniChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        self._backend_id = srsly.read_json(path)["backend_id"]
        self._backend = getattr(minichain.backend, self._backend_id)

        return self
