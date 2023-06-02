# mypy: ignore-errors
import pytest
from typing import Iterable

from spacy.tokens import Doc

from ...registry import registry
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
    pass
