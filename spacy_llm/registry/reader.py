from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Union, cast

import srsly

from .util import registry


@registry.misc("spacy.JinjaReader.v1")
def jinja_reader(path: Union[str, Path]) -> str:
    """Read a jinja2 file containing a prompt template

    path (Union[str, Path]): path to a .jinja2 template file

    RETURNS (jinja2.Template): a jinja2 Template that can be rendered
    """
    tpl_path = Path(path) if isinstance(path, str) else path
    if not tpl_path.suffix == ".jinja2":
        raise ValueError("The JinjaReader expects a .jinja2 file.")

    with tpl_path.open("r", encoding="utf8") as tpl_file:
        tpl_text = tpl_file.read()
    return tpl_text


@registry.misc("spacy.FewShotReader.v1")
def fewshot_reader(path: Union[str, Path]) -> Callable[[], Iterable[Dict[str, Any]]]:
    """Read a file containing examples to include in few-shot learning

    path (Union[str,Path]): path to an examples file (.yml, .yaml, .json, .jsonl)

    RETURNS (Iterable[Dict[str, Any]]): an iterable of examples to be parsed by the template
    """

    # typecast string path so that we can use pathlib functionality
    eg_path = Path(path) if isinstance(path, str) else path

    def reader() -> Iterable[Dict[str, Any]]:
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
        return cast(Iterable[Dict[str, Any]], data)

    return reader
