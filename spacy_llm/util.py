from typing import Any, Dict, Union
from pathlib import Path
from confection import Config

from spacy.util import SimpleFrozenDict, get_sourced_components, load_config
from spacy.util import load_model_from_config


def assemble_from_config(
    config=Config,
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
