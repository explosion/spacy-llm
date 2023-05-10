from pathlib import Path
from typing import Any, Callable, Iterable, Union

import srsly

from .util import registry


@registry.misc("spacy.FewShotReader.v1")
def fewshot_reader(path: Union[str, Path]) -> Callable[[], Iterable[Any]]:
    """Read a file containing examples to include in few-shot learning

    path (Union[str,Path]): path to an examples file (.yml, .yaml, .json, .jsonl)

    RETURNS (Iterable[Any]): an iterable of examples to be parsed by the template
    """

    # typecast string path so that we can use pathlib functionality
    eg_path = Path(path) if isinstance(path, str) else path

    def reader() -> Iterable[Any]:
        if eg_path is None:
            data = []
        elif eg_path.suffix in (".yml", ".yaml"):
            data = srsly.read_yaml(eg_path)
        elif eg_path.suffix == ".json":
            data = srsly.read_json(eg_path)
        elif eg_path.suffix == ".jsonl":
            data = list(srsly.read_jsonl(eg_path))
        else:
            raise ValueError(
                "The examples file expects a .yml, .yaml, .json, or .jsonl file type."
            )

        if not isinstance(data, list):
            raise ValueError(
                f"Cannot interpret prompt examples from {path}. Please check your formatting"
            )
        return data

    return reader
