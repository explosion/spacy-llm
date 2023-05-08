import spacy

BACKEND_CFG = {
    "backend": {
        "@llm_backends": "spacy.MiniChain.v1",
        "api": "OpenAI",
        "config": {},
        "query": {"@llm_queries": "spacy.RunMiniChain.v1"},
    }
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=BACKEND_CFG)
    nlp("This is a test.")
