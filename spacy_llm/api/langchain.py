from pathlib import Path
from typing import Callable, Iterable, Any, Tuple, cast, Union, Dict

import spacy
import srsly
import langchain

from spacy.util import SimpleFrozenList


class LangChain:
    def __init__(
        self,
        backend: str,
        prompt: Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]],
        backend_config: Dict[Any, Any],
    ):
        """Initialize wrapper for LangChain.
        backend (str): Name of any entry in langchain.llms.type_to_cls_dict, e. g. 'openai'.
        prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
        backend_config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the
            langchain.llms.BaseLLM instance.
        """
        self._backend_id = backend
        self._backend: langchain.llms.BaseLLM = langchain.llms.type_to_cls_dict[
            backend
        ](**backend_config)
        self._prompt = prompt
        self._backend_config = backend_config

    def prompt(self, prompts: Iterable[Any]) -> Iterable[Any]:
        return self._prompt(self._backend, prompts)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps(
            {"backend": self._backend_id, "llm_config": self._backend_config}
        )

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LangChain":
        data = srsly.msgpack_loads(bytes_data)
        self._backend_id = data["backend"]
        self._backend_config = data["llm_config"]
        self._backend = langchain.llms.type_to_cls_dict[self._backend_id](
            **self._backend_config
        )

        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(
            path, {"backend": self._backend_id, "llm_config": self._backend_config}
        )

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "LangChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        data = srsly.read_json(path)
        self._backend_id = data["backend_id"]
        self._backend_config = data["llm_config"]
        self._backend = langchain.llms.type_to_cls_dict[self._backend_id](
            **self._backend_config
        )

        return self
