import abc
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type, TypeVar, cast

import srsly
from pydantic import BaseModel
from spacy import util

ExampleType = TypeVar("ExampleType", bound=BaseModel)


class SerializableTask(abc.ABC, Generic[ExampleType]):
    """A task that can be serialized and deserialized."""

    _examples: Optional[List[ExampleType]]

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

    def serialize_cfg(self) -> bytes:
        """Serialize the task's configuration attributes."""
        cfg = {key: getattr(self, key) for key in self._cfg_keys}
        return srsly.json_dumps(cfg)

    def deserialize_cfg(self, b: bytes) -> None:
        """Deserialize the task's configuration attributes.

        b (bytes): serialized configuration attributes.
        """
        cfg = srsly.json_loads(b)
        for key, value in cfg.items():
            setattr(self, key, value)

    def serialize_examples(self) -> bytes:
        """Serialize examples."""
        if self._examples is None:
            return srsly.json_dumps(None)
        examples = [eg.dict() for eg in self._examples]
        return srsly.json_dumps(examples)

    def deserialize_examples(self, b: bytes) -> None:
        """Deserialize examples from bytes.

        b (bytes): serialized examples.
        """
        examples = srsly.json_loads(b)
        if examples is not None:
            self._examples = [self._Example.parse_obj(eg) for eg in examples]

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
            "cfg": self.serialize_cfg,
            "examples": self.serialize_examples,
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
            "cfg": lambda b: self.deserialize_cfg(b),
            "examples": lambda b: self.deserialize_examples(b),
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
            "cfg": lambda p: srsly.write_json(p, self.serialize_cfg()),
            "examples.json": lambda p: p.write_text(self.serialize_examples()),
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
            "cfg.json": lambda p: self.deserialize_cfg(p.read_text()),
            "examples.json": lambda p: self.deserialize_examples(p.read_text()),
        }

        util.from_disk(path, deserialize, exclude)
        return self
