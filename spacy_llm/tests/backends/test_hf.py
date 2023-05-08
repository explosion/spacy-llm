import spacy

BACKEND_CFG = {
    "backend": {
        "@llm_backends": "spacy.HF.v1",
        "model": "databricks/dolly-v2-12b",
    },
}


def test_integrations():
    """Test simple run."""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=BACKEND_CFG)
    nlp("This is a test.")
