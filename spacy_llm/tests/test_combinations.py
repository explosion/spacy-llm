import copy
from typing import Any, Dict

import pytest
import spacy
from thinc.api import NumpyOps, get_current_ops

from spacy_llm.compat import has_langchain, has_minichain
from spacy_llm.pipeline import LLMWrapper

PIPE_CFG: Dict[str, Any] = {
    "model": {
        "@llm_models": None,
        "config": {},
    },
    "task": {"@llm_tasks": None},
}


@pytest.mark.external
@pytest.mark.skipif(has_minichain is False, reason="MiniChain is not installed")
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
@pytest.mark.parametrize(
    "model",
    ["spacy.LangChain.v1", "spacy.MiniChain.v1", "spacy.OpenAI.gpt-3.5.v1"],
    ids=["langchain", "minichain", "rest-openai"],
)
@pytest.mark.parametrize(
    "task",
    ["spacy.NER.v1", "spacy.NER.v2", "spacy.TextCat.v1"],
    ids=["ner.v1", "ner.v2", "textcat"],
)
@pytest.mark.parametrize("n_process", [1, 2])
def test_combinations(model: str, task: str, n_process: int):
    """Randomly test combinations of models and tasks."""
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")

    config = copy.deepcopy(PIPE_CFG)
    config["model"]["@llm_models"] = model
    if "MiniChain" in model:
        config["model"]["config"] = {"model": "gpt-3.5-turbo"}
        config["model"]["api"] = "OpenAI"
    elif "LangChain" in model:
        config["model"]["config"] = {"model": "ada"}
        config["model"]["api"] = "OpenAI"
    config["task"]["@llm_tasks"] = task

    # Configure task-specific settings.
    if task.startswith("spacy.NER"):
        config["task"]["labels"] = "PER,ORG,LOC"
    elif task.startswith("spacy.TextCat"):
        config["task"]["labels"] = "Recipe"
        config["task"]["exclusive_classes"] = True

    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=config)
    name, component = nlp.pipeline[0]
    assert name == "llm"
    assert isinstance(component, LLMWrapper)

    nlp("This is a test.")
    list(
        nlp.pipe(["This is a second test", "This is a third test"], n_process=n_process)
    )
