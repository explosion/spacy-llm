import functools
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

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


def _fewshot_reader(eg_path: Path) -> Iterable[Dict[str, Any]]:
    data: Optional[List] = None

    if eg_path is None:
        data = []

    else:
        suffix = eg_path.suffix.replace("yaml", "yml")
        readers = {
            ".yml": srsly.read_yaml,
            ".json": srsly.read_json,
            ".jsonl": lambda path: list(srsly.read_jsonl(eg_path)),
        }

        # Sort formats/read methods depending on file suffix so that the read methods most likely to work are used
        # first.
        if suffix == ".json":
            formats = (".json", ".jsonl", ".yml")
        elif suffix == ".jsonl":
            formats = (".jsonl", ".json", ".yml")
        else:
            formats = (".yml", ".json", ".jsonl")

        # Try to read file in all supported formats.
        i = 0
        for i, file_format in enumerate(formats):
            try:
                data = readers[file_format](eg_path)
                break
            except Exception:
                pass

        # Raise error if reading file didn't work.
        if data is None:
            raise ValueError(
                "The examples file expects a .yml, .yaml, .json, or .jsonl file type. Ensure that your file "
                "corresponds to one of these file formats."
            )

        # Reading worked, but suffix is wrong: recommend changing suffix.
        if i > 0 or suffix not in formats:
            warnings.warn(
                "Content of examples file could be read, but the file suffix does not correspond to the detected "
                "format. Please ensure the correct suffix has been used."
            )

    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError(
            f"Cannot interpret prompt examples from {str(eg_path)}. Please check your formatting."
        )

    return cast(Iterable[Dict[str, Any]], data)
