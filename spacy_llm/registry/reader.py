import functools
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Union, cast

import srsly


def file_reader(path: Union[str, Path]) -> str:
    """Read a file from a path and return its contents.

    path (Union[str, Path]): path to the file

    RETURNS (str): the contents of the file
    """
    tpl_path = Path(path) if isinstance(path, str) else path
    with tpl_path.open("r", encoding="utf8") as tpl_file:
        tpl_text = tpl_file.read()

    return tpl_text


def fewshot_reader(path: Union[str, Path]) -> Callable[[], Iterable[Dict[str, Any]]]:
    """Read a file containing examples to include in few-shot learning

    path (Union[str,Path]): path to an examples file (.yml, .yaml, .json, .jsonl)

    RETURNS (Iterable[Dict[str, Any]]): an iterable of examples to be parsed by the template
    """
    eg_path = Path(path) if isinstance(path, str) else path
    return functools.partial(_fewshot_reader, eg_path=eg_path)


def _fewshot_reader(eg_path: Path) -> Iterable[Any]:
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

    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError(
            f"Cannot interpret prompt examples from {str(eg_path)}. Please check your formatting"
        )
    return cast(Iterable[Dict[str, Any]], data)
