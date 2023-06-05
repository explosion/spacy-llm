# mypy: ignore-errors
import pytest
import copy

import spacy

from ..compat import has_anthropic_key

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.REST.v1",
        "api": "Anthropic",
        "config": {"temperature": 0.3, "model": "claude-v1"},
    },
    "task": {"@llm_tasks": "spacy.TextCat.v1", "labels": "POSITIVE,NEGATIVE"},
}


@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
@pytest.mark.external
def test_model_error_handling():
    """Test error handling for wrong model"""
    nlp = spacy.blank("en")
    with pytest.raises(ValueError) as err:
        cfg = copy.deepcopy(PIPE_CFG)
        cfg["backend"]["config"] = {"model": "x-gpt-3.5-turbo"}
        nlp.add_pipe("llm", config=cfg)

    assert "The specified model 'x-gpt-3.5-turbo' is not available." in str(err.value)
