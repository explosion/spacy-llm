from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import spacy
import srsly

for registry_name in ("queries", "backends", "tasks"):
    if f"llm_{registry_name}" not in spacy.registry.get_registry_names():
        spacy.registry.create(f"llm_{registry_name}", entry_points=True)


@spacy.registry.misc("spacy.ExampleReader.v1")
def example_reader(path: Optional[str] = None) -> Callable[[str], Iterable[Any]]:
    """Read an examples file to include in few-shot learning

    path (Path): path to an examples file (.yml, .yaml, .jsonl)

    RETURNS (Iterable[Any]): an iterable of examples to be parsed by the template
    """

    # typecast string path so that we can use pathlib functionality
    path = Path(path)

    def reader() -> Iterable[Any]:
        if path is None:
            data = []
        elif path.suffix in (".yml", ".yaml"):
            data = srsly.read_yaml(path)
        elif path.suffix == ".jsonl":
            data = srsly.read_jsonl(path)
        else:
            raise ValueError(
                "The examples file expects a .yml, .yaml, or .jsonl file type."
            )

        if not isinstance(data, list):
            raise ValueError(
                f"Cannot interpret prompt examples from {path}. Please check your formatting"
            )
        return data

    return reader
