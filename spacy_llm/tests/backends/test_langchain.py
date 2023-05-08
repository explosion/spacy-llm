import spacy

BACKEND_CFG = {
    "backend": {
        "@llm_backends": "spacy.LangChain.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3},
        "query": {"@llm_queries": "spacy.CallLangChain.v1"},
    },
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=BACKEND_CFG)
    nlp("This is a test.")
