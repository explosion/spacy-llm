import pytest
import spacy

from spacy_llm.compat import has_langchain

PIPE_CFG = {
    "model": {
        "@llm_models": "langchain.OpenAI.v1",
        "name": "ada",
        "config": {"temperature": 0.3},
        "query": {"@llm_queries": "spacy.CallLangChain.v1"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


@pytest.mark.external
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


@pytest.mark.external
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
def test_initialization_azure_openai():
    """Test initialization and simple run with Azure OpenAI models."""
    pipe_cfg = {
        "model": {
            "@llm_models": "langchain.Azure.v1",
            "name": "ada",
            "config": {"temperature": 0.3},
            "query": {"@llm_queries": "spacy.CallLangChain.v1"},
        },
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
    }

    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=pipe_cfg)
    nlp("This is a test.")
