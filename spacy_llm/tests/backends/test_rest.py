# mypy: ignore-errors
import copy

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


@pytest.mark.external
def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


@pytest.mark.external
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


@pytest.mark.external
def test_openai_chat():
    """Test OpenAI call to /chat backend (/completions backend is in PIPE_CFG and tested in other tests)."""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["config"]["model"] = "gpt-3.5-turbo"
    cfg["backend"]["config"]["url"] = "https://api.openai.com/v1/chat/completions"
    nlp.add_pipe(
        "llm",
        config=cfg,
    )
    nlp("test")


def test_model_backend_compatibility():
    """Tests whether incompatible model and backend are detected as expected."""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["config"]["model"] = "gpt-4"
    with pytest.raises(
        ValueError,
        match="Specified model gpt-4 supports of the following endpoints: "
        "https://api.openai.com/v1/chat/completions. However, endpoint https://api.openai.com/v1/completions "
        "has been configured. Please ensure that model and endpoint match.",
    ):
        nlp.add_pipe(
            "llm",
            config=cfg,
        )
