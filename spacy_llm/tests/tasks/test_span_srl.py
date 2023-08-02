import pytest

from confection import Config
from pathlib import Path
from pytest import FixtureRequest
from spacy_llm.pipeline import LLMWrapper
from spacy_llm.tasks.srl_task import SRLExample
from spacy_llm.tests.compat import has_openai_key
from spacy_llm.ty import Labeled, LLMTask
from spacy_llm.util import assemble_from_config, split_labels

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
    labels = ARG-0,ARG-1,ARG-2,ARG-M-LOC,ARG-M-TMP
    
    [components.llm.task.label_definitions]
    ARG-0 = "Agent"
    ARG-1 = "Patient or Theme"
    ARG-2 = "ARG-2"
    ARG-M-TMP = "Temporal Modifier"
    ARG-M-LOC = "Location Modifier"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def task():
    text = "We love this sentence in Berlin right now ."
    predicate = {"text": "love", "start_char": 3, "end_char": 7}
    srl_example = SRLExample(
        **{
            "text": text,
            "predicates": [predicate],
            "relations": [
                (
                    predicate,
                    [
                        {
                            "label": "ARG-0",
                            "role": {"text": "We", "start_char": 0, "end_char": 2},
                        },
                        {
                            "label": "ARG-1",
                            "role": {
                                "text": "this sentence",
                                "start_char": 8,
                                "end_char": 21,
                            },
                        },
                        {
                            "label": "ARG-M-LOC",
                            "role": {
                                "text": "in Berlin",
                                "start_char": 22,
                                "end_char": 31,
                            },
                        },
                        {
                            "label": "ARG-M-TMP",
                            "role": {
                                "text": "right now",
                                "start_char": 32,
                                "end_char": 41,
                            },
                        },
                    ],
                )
            ],
        }
    )
    return text, srl_example


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
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string"])
def test_rel_predict(task, cfg_string, request):
    """Use OpenAI to get REL results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)

    text, gold_example = task
    doc = nlp(text)

    assert len(doc._.predicates)
    assert len(doc._.relations)

    assert doc._.predicates[0]["text"] == gold_example.predicates[0].text

    predicated_roles = tuple(
        sorted([r["role"]["text"] for p, rs in doc._.relations for r in rs])
    )
    gold_roles = tuple(
        sorted([r.role.text for p, rs in gold_example.relations for r in rs])
    )

    assert predicated_roles == gold_roles
