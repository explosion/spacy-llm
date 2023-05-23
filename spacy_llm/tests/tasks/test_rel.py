from pathlib import Path

import pytest
import spacy
from confection import Config
from pytest import FixtureRequest

from spacy_llm.tasks.rel import RelationItem

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [nlp]
    lang = "en"
    pipeline = ["ner", "llm"]
    batch_size = 128

    [components]

    [components.ner]
    source = "en_core_web_md"

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.REL.v1"
    labels = "LivesIn,Visits"

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"
    """


@pytest.fixture
def fewshot_cfg_string():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["ner", "llm"]
    batch_size = 128

    [components]

    [components.ner]
    source = "en_core_web_md"

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.REL.v1"
    labels = "LivesIn,Visits"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str(EXAMPLES_DIR / "rel_examples.jsonl")}

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"
    """


@pytest.fixture
def task():
    text = "Joey rents a place in New York City."
    labels = "LivesIn,Visits"
    gold_relations = [RelationItem(dep=0, dest=1, relation="LivesIn")]
    examples_path = str(EXAMPLES_DIR / "rel_examples.jsonl")
    return text, labels, gold_relations, examples_path


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_rel_config(cfg_string, request: FixtureRequest):
    """Simple test to check if the config loads properly given different settings"""

    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
