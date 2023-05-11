import copy
import itertools

import pytest
import spacy

from spacy_llm.compat import has_minichain, has_langchain

PIPE_CFG = {
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
def test_combinations():
    """Randomly test combinations of backends and tasks."""
    backends = ("spacy.LangChain.v1", "spacy.MiniChain.v1", "spacy.REST.v1")
    tasks = ("spacy.NER.v1", "spacy.TextCat.v1")

    for combination in list(itertools.product(backends, tasks)):
        config = copy.deepcopy(PIPE_CFG)
        config["backend"]["@llm_backends"] = combination[0]
        config["task"]["@llm_tasks"] = combination[1]

        # Configure task-specific settings.
        if combination[1].startswith("spacy.NER"):
            config["task"]["labels"] = "PER,ORG,LOC"
        elif combination[1].startswith("spacy.TextCat"):
            config["task"]["labels"] = "Recipe"
            config["task"]["exclusive_classes"] = True

        nlp = spacy.blank("en")
        nlp.add_pipe("llm", config=config)
        nlp("This is a test.")
