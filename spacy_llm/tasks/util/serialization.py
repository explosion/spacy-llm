import abc
from pathlib import Path
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, cast

import srsly
from pydantic import BaseModel
from spacy import util

ExampleType = TypeVar("ExampleType", bound=BaseModel)


class SerializableTask(abc.ABC, Generic[ExampleType]):
    """A task that can be serialized and deserialized."""

    _prompt_examples: List[ExampleType]

    @property
    @abc.abstractmethod
    def _cfg_keys(self) -> List[str]:
        """A list of configuration attributes to serialize."""
        pass

    @property
    @abc.abstractmethod
    def _Example(self) -> Type[ExampleType]:
        """The example type."""
        pass

    def get_cfg(self) -> Dict[str, Any]:
        """Serialize the task's configuration attributes."""
        cfg = {key: getattr(self, key) for key in self._cfg_keys}
        return cfg

    def set_cfg(self, cfg: Dict[str, Any]) -> None:
        """Deserialize the task's configuration attributes.

        cfg (Dict[str, Any]): dictionary containing configuration attributes.
        """
        for key, value in cfg.items():
            setattr(self, key, value)

    def _get_prompt_examples(self) -> List[Dict[str, Any]]:
        """Serialize examples."""
        examples = [eg.dict() for eg in self._prompt_examples]
        return examples

    def _set_prompt_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Deserialize examples from bytes.

        examples (List[Dict[str, Any]]): serialized examples.
        """
        self._prompt_examples = [self._Example.parse_obj(eg) for eg in examples]

    def to_bytes(
        self,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """

        serialize = {
            "cfg": lambda: srsly.json_dumps(self.get_cfg()),
            "prompt_examples": lambda: srsly.msgpack_dumps(self._get_prompt_examples()),
        }

        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "SerializableTask":
        """Load the Task from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple[str]): Names of properties to exclude from deserialization.
        RETURNS (SpanTask): Modified SpanTask instance.
        """

        deserialize = {
            "cfg": lambda b: self.set_cfg(srsly.json_loads(b)),
            "prompt_examples": lambda b: self._set_prompt_examples(
                srsly.msgpack_loads(b)
            ),
        }

        util.from_bytes(bytes_data, deserialize, exclude)
        return self

    def to_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> None:
        """Serialize the task to disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """

        serialize = {
            "cfg": lambda p: srsly.write_json(p, self.get_cfg()),
            "prompt_examples": lambda p: srsly.write_msgpack(
                p, self._get_prompt_examples()
            ),
        }

        util.to_disk(path, serialize, exclude)

    def from_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "SerializableTask":
        """Deserialize the task from disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """

        deserialize = {
            "cfg": lambda p: self.set_cfg(srsly.read_json(p)),
            "prompt_examples": lambda p: self._set_prompt_examples(
                srsly.read_msgpack(p)
            ),
        }

        util.from_disk(path, deserialize, exclude)
        return self
