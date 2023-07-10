import functools
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
        if not eg_path.exists():
            raise ValueError(
                "Specified file path doesn't exist. Please ensure to provide a valid file path."
            )

        suffix = eg_path.suffix.replace("yaml", "yml")
        readers = {
            ".yml": srsly.read_yaml,
            ".json": srsly.read_json,
            ".jsonl": lambda path: list(srsly.read_jsonl(eg_path)),
        }

        # Try to read in indicated format.
        success = False
        if suffix in readers:
            try:
                data = readers[suffix](eg_path)
                success = True
            except Exception:
                pass
        if not success:
            # Try to read file in all supported formats.
            for file_format, reader in readers.items():
                if file_format == suffix:
                    continue
                try:
                    data = reader(eg_path)
                    success = True
                    break
                except Exception:
                    pass

        # Raise error if reading file didn't work.
        if not success:
            raise ValueError(
                "The examples file expects a .yml, .yaml, .json, or .jsonl file type. Ensure that your file "
                "corresponds to one of these file formats."
            )

    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError(
            f"Cannot interpret prompt examples from {str(eg_path)}. Please check your formatting to ensure that the "
            f"examples specified in {eg_path} are described as list of dictionaries that fit the structure described by"
            f" the prompt example class for the corresponding class."
        )

    return cast(Iterable[Dict[str, Any]], data)
