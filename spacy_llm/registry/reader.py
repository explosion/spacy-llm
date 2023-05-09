from pathlib import Path
from typing import Any, Callable, Iterable

import spacy
import srsly


@spacy.registry.misc("spacy.ExampleReader.v1")
def example_reader(path: str) -> Callable[[], Iterable[Any]]:
    """Read an examples file to include in few-shot learning

    path (Path): path to an examples file (.yml, .yaml, .json)

    RETURNS (Iterable[Any]): an iterable of examples to be parsed by the template
    """

    # typecast string path so that we can use pathlib functionality
    eg_path = Path(path)

    def reader() -> Iterable[Any]:
        if eg_path is None:
            data = []
        elif eg_path.suffix in (".yml", ".yaml"):
            data = srsly.read_yaml(eg_path)
        elif eg_path.suffix == ".json":
            data = srsly.read_json(eg_path)
        else:
            raise ValueError(
                "The examples file expects a .yml, .yaml, or .json file type."
            )

        if not isinstance(data, list):
            raise ValueError(
                f"Cannot interpret prompt examples from {path}. Please check your formatting"
            )
        return data

    return reader
