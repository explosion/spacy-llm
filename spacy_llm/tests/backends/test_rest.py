# mypy: ignore-errors
import pytest
import spacy

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.REST.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3, "model": "text-davinci-003"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


def test_rest_backend_error_handling():
    """Test error handling for default/minimal REST backend."""
    nlp = spacy.blank("en")
    with pytest.raises(ValueError) as err:
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "backend": {"config": {"model": "x-text-davinci-003"}},
            },
        )
    assert "The specified model 'x-text-davinci-003' is not available." in str(
        err.value
    )
