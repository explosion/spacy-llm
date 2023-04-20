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
        prompt: Callable[["minichain.Backend", Iterable[str]], Iterable[str]],
        backend_config: Dict[Any, Any],
    ):
        """Initialize wrapper for MiniChain.
        backend (str): Name of any backend class in minichain.backend, e. g. "OpenAI".
        prompt (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]): Callable executing prompts.
        backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the
            minichain.Backend instance.
        """
        self._backend_id = backend
        self._backend_config = backend_config
        self._backend = self._load_backend()
        self._prompt = prompt

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        return self._prompt(self._backend, prompts)

    def _load_backend(self) -> "minichain.Backend":
        """Loads MiniChain backend.
        RETURNS (minichain.Backend): Loaded backend
        """
        if hasattr(minichain.backend, self._backend_id):
            return getattr(minichain.backend, self._backend_id)(**self._backend_config)
        else:
            raise KeyError(
                f"The requested backend {self._backend_id} is not available in `minichain.backend`."
            )

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps(
            {"backend_id": self._backend_id, "backend_config": self._backend_config}
        )

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "MiniChain":
        data = srsly.msgpack_loads(bytes_data)
        self._backend_id = data["backend_id"]
        self._backend_config = data["backend_config"]
        self._backend = self._load_backend()

        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(
            path,
            {"backend_id": self._backend_id, "backend_config": self._backend_config},
        )

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "MiniChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        data = srsly.read_json(path)
        self._backend_id = data["backend_id"]
        self._backend_config = data["backend_config"]
        self._backend = self._load_backend()

        return self
