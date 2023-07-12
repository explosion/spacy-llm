import re
from pathlib import Path
from typing import Callable

import pytest
from confection import Config

from spacy_llm.registry import registry
from spacy_llm.util import assemble_from_config

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@pytest.fixture
def ner_noop_config_str() -> str:
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v2"
    labels = ["PERSON", "LOCATION"]
    
    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    """


def test_normalizer_registry_resolution(ner_noop_config_str):
    """Test whether registry resolution of normalizer handle is done correctly."""
    config = Config().from_str(ner_noop_config_str)
    config["components"]["llm"]["task"].pop("normalizer")
    assemble_from_config(config)

    def _strip(text: str) -> int:
        return 0

    @registry.misc("InvalidNormalizer.v1")
    def normalizer_factory() -> Callable[[str], int]:
        return _strip

    config["components"]["llm"]["task"]["normalizer"] = "InvalidNormalizer.v1"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`normalizer` has to be of type typing.Callable[[str], str], but is of type {'return': "
            "typing.Callable[[str], int]}."
        ),
    ):
        assemble_from_config(config)
