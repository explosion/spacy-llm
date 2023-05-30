import spacy
import pytest
from spacy_llm.compat import has_minichain

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.MiniChain.v1",
        "api": "OpenAI",
        "config": {"model": "gpt-3.5-turbo"},
        "query": {"@llm_queries": "spacy.RunMiniChain.v1"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


@pytest.mark.external
@pytest.mark.skipif(has_minichain is False, reason="MiniChain is not installed")
def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")
