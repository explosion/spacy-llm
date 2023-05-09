from pathlib import Path
from typing import Any, Iterable

import spacy
import srsly

for registry_name in ("queries", "backends", "tasks"):
    if f"llm_{registry_name}" not in spacy.registry.get_registry_names():
        spacy.registry.create(f"llm_{registry_name}", entry_points=True)


def read_examples_file(path: Path) -> Iterable[Any]:
    """Read an examples file to include in few-shot learning

    path (Path): path to an examples file (.yml, .yaml, .jsonl)

    RETURNS (Iterable[Any]): an iterable of examples to be parsed by the template
    """
    if path.suffix in (".yml", ".yaml"):
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
