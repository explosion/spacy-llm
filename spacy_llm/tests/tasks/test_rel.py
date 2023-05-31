from pathlib import Path

import pytest
from confection import Config
from pytest import FixtureRequest

from spacy_llm.tasks.rel import RelationItem
from spacy_llm.util import assemble_from_config

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

    [initialize]
    vectors = "en_core_web_md"
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
    labels = ["LivesIn", "Visits"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str(EXAMPLES_DIR / "rel_examples.jsonl")}

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"

    [initialize]
    vectors = "en_core_web_md"
    """


@pytest.fixture
def task():
    text = "Joey rents a place in New York City."
    gold_relations = [RelationItem(dep=0, dest=1, relation="LivesIn")]
    return text, gold_relations


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_rel_config(cfg_string, request: FixtureRequest):
    """Simple test to check if the config loads properly given different settings"""

    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)
    assert nlp.pipe_names == ["ner", "llm"]


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_rel_predict(task, cfg_string, request):
    """Use OpenAI to get REL results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)

    text, _ = task
    doc = nlp(text)

    assert doc.ents
    assert doc._.rel
