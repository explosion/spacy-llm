from pathlib import Path

import pytest
from confection import Config
from pytest import FixtureRequest
from spacy_llm.pipeline import LLMWrapper
from spacy_llm.ty import Labeled, LLMTask
from spacy_llm.util import assemble_from_config, split_labels

from spacy_llm.tests.compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [paths]
    examples = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.SRL.v1"
    labels = ARG-0,ARG-1,ARG-M-TMP,ARG-M-LOC

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def task():
    text = "We love this sentence in Berlin right now ."
    gold_relations = []
    return text, gold_relations


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string"])
def test_rel_config(cfg_string, request: FixtureRequest):
    """Simple test to check if the config loads properly given different settings"""
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)
    assert nlp.pipe_names == ["llm"]

    pipe = nlp.get_pipe("llm")
    assert isinstance(pipe, LLMWrapper)
    assert isinstance(pipe.task, LLMTask)

    task = pipe.task
    labels = orig_config["components"]["llm"]["task"]["labels"]
    labels = split_labels(labels)
    assert isinstance(task, Labeled)
    assert task.labels == tuple(labels)
    assert set(pipe.labels) == set(task.labels)
    assert nlp.pipe_labels["llm"] == list(task.labels)


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string", ["zeroshot_cfg_string"]
)  # "zeroshot_cfg_string",
def test_rel_predict(task, cfg_string, request):
    """Use OpenAI to get REL results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)

    text, _ = task
    doc = nlp(text)

    assert doc._.predicates
    assert doc._.relations
