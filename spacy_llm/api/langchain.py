from pathlib import Path
from typing import Callable, Iterable, Any, Tuple, cast, Union, Dict

import spacy
import srsly
from spacy.util import SimpleFrozenList

from ..compat import langchain


class LangChain:
    def __init__(
        self,
        backend: str,
        prompt: Callable[["langchain.llms.BaseLLM", Iterable[Any]], Iterable[Any]],
        backend_config: Dict[Any, Any],
    ):
        """Initialize wrapper for LangChain.
        backend (str): Name of any entry in langchain.llms.type_to_cls_dict, e. g. 'openai'.
        prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
        backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the
            langchain.llms.BaseLLM instance.
        """
        self._backend_id = backend
        self._backend_config = backend_config
        self._backend = self._load_backend()
        self._prompt = prompt

    def _load_backend(self) -> "langchain.llms.BaseLLM":
        """Loads LangChain backend.
        RETURNS (langchain.llms.BaseLLM): LangChain backend.
        """
        try:
            return langchain.llms.type_to_cls_dict[self._backend_id](
                **self._backend_config
            )
        except KeyError as err:
            raise KeyError(
                f"The requested backend {self._backend_id} is not available in `langchain.llms.type_to_cls_dict`."
            ) from err

    def __call__(self, prompts: Iterable[Any]) -> Iterable[Any]:
        return self._prompt(self._backend, prompts)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps(
            {"backend_id": self._backend_id, "llm_config": self._backend_config}
        )

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LangChain":
        data = srsly.msgpack_loads(bytes_data)
        self._backend_id = data["backend_id"]
        self._backend_config = data["llm_config"]
        self._backend = self._load_backend()

        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(
            path, {"backend_id": self._backend_id, "llm_config": self._backend_config}
        )

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "LangChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        data = srsly.read_json(path)
        self._backend_id = data["backend_id"]
        self._backend_config = data["llm_config"]
        self._backend = self._load_backend()

        return self
