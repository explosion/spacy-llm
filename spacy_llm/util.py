from typing import Any, Dict, Union
from pathlib import Path
from confection import Config

from spacy.util import SimpleFrozenDict, get_sourced_components, load_config
from spacy.util import load_model_from_config


def assemble(
    config_or_path: Union[str, Path, Config],
    *,
    overrides: Dict[str, Any] = SimpleFrozenDict(),
):
    config: Config

    if isinstance(config_or_path, Config):
        config = config_or_path
    else:
        config_or_path = Path(config_or_path)
        config = load_config(config_or_path, overrides=overrides, interpolate=False)

    nlp = load_model_from_config(config, auto_fill=True)
    config = config.interpolate()
    sourced = get_sourced_components(config)
    nlp._link_components()
    with nlp.select_pipes(disable=[*sourced]):
        nlp.initialize()
    return nlp
