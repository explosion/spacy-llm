from pathlib import Path
from typing import Callable, Iterable, Any, Tuple, cast, Union

import spacy
import srsly
import langchain

from spacy.util import SimpleFrozenList


class LangChain:
    def __init__(
        self,
        backend: str,
        prompt: Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]],
        **kwargs
    ):
        """Initialize wrapper for LangChain. All kwargs will be passed on to the initialization of the LangChain LLM
        object.
        backend (str): Name of any entry in langchain.llms.type_to_cls_dict, e. g. 'openai'.
        prompt (Callable[[langchain.llms.BaseLLM, Iterable[Any]], Iterable[Any]]): Callable executing prompts.
        """
        self._backend_id = backend
        self._backend: langchain.llms.BaseLLM = langchain.llms.type_to_cls_dict[
            backend
        ](**kwargs)
        self._prompt = prompt
        self._llm_config = kwargs

    def prompt(self, prompts: Iterable[Any]) -> Iterable[Any]:
        return self._prompt(self._backend, prompts)

    def to_bytes(self, *, exclude: Tuple[str] = cast(Tuple[str], tuple())) -> bytes:
        return srsly.msgpack_dumps(
            {"backend": self._backend_id, "llm_config": self._llm_config}
        )

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LangChain":
        data = srsly.msgpack_loads(bytes_data)
        self._backend_id = data["backend"]
        self._llm_config = data["llm_config"]
        self._backend = langchain.llms.type_to_cls_dict[self._backend_id](
            **self._llm_config
        )

        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        path = spacy.util.ensure_path(path).with_suffix(".json")
        srsly.write_json(
            path, {"backend": self._backend_id, "llm_config": self._llm_config}
        )

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "LangChain":
        path = spacy.util.ensure_path(path).with_suffix(".json")
        data = srsly.read_json(path)
        self._backend_id = data["backend_id"]
        self._llm_config = data["llm_config"]
        self._backend = langchain.llms.type_to_cls_dict[self._backend_id](
            **self._llm_config
        )

        return self
