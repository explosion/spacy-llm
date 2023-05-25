from typing import Any, Dict, Iterable, List, Union
from pathlib import Path

from confection import Config
from spacy.util import SimpleFrozenDict, get_sourced_components, load_config
from spacy.util import load_model_from_config


def split_labels(labels: Union[str, Iterable[str]]) -> List[str]:
    """Split a comma-separated list of labels.
    If input is a list already, just strip each entry of the list

    labels (Union[str, Iterable[str]]): comma-separated string or list of labels
    RETURNS (List[str]): a split and stripped list of labels
    """
    labels = labels.split(",") if isinstance(labels, str) else labels
    return [label.strip() for label in labels]


def assemble_from_config(
    config: Config,
):
    nlp = load_model_from_config(config, auto_fill=True)
    config = config.interpolate()
    sourced = get_sourced_components(config)
    nlp._link_components()
    with nlp.select_pipes(disable=[*sourced]):
        nlp.initialize()
    return nlp


def assemble(
    config_path: Union[str, Path],
    *,
    overrides: Dict[str, Any] = SimpleFrozenDict(),
):
    config_path = Path(config_path)
    config = load_config(config_path, overrides=overrides, interpolate=False)
    nlp = load_model_from_config(config, auto_fill=True)
    config = config.interpolate()
    sourced = get_sourced_components(config)
    nlp._link_components()
    with nlp.select_pipes(disable=[*sourced]):
        nlp.initialize()
    return nlp
