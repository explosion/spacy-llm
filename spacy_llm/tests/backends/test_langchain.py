import spacy

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.LangChain.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3},
        "query": {"@llm_queries": "spacy.CallLangChain.v1"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")
