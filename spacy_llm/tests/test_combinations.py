import copy
from typing import Any, Dict

import pytest
import spacy
from thinc.api import get_current_ops, NumpyOps

from spacy_llm.compat import has_minichain, has_langchain
from spacy_llm.pipeline import LLMWrapper

PIPE_CFG: Dict[str, Any] = {
    "backend": {
        "@llm_backends": None,
        "api": "OpenAI",
        "config": {},
    },
    "task": {"@llm_tasks": None},
}


@pytest.mark.external
@pytest.mark.skipif(has_minichain is False, reason="MiniChain is not installed")
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
@pytest.mark.parametrize(
    "backend",
    ["spacy.LangChain.v1", "spacy.MiniChain.v1", "spacy.REST.v1"],
    ids=["langchain", "minichain", "rest"],
)
@pytest.mark.parametrize(
    "task",
    ["spacy.NER.v1", "spacy.NER.v2", "spacy.TextCat.v1"],
    ids=["ner.v1", "ner.v2", "textcat"],
)
@pytest.mark.parametrize("n_process", [1, 2])
def test_combinations(backend: str, task: str, n_process: int):
    """Randomly test combinations of backends and tasks."""
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")

    config = copy.deepcopy(PIPE_CFG)
    config["backend"]["@llm_backends"] = backend
    config["backend"]["config"] = {
        "model": "ada" if backend != "spacy.MiniChain.v1" else "gpt-3.5-turbo"
    }
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
