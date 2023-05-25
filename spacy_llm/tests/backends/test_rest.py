# mypy: ignore-errors
import copy
import re
from typing import Iterable

import pytest
import spacy
from spacy.tokens import Doc

from ...registry import registry
from ..compat import has_openai_key

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.REST.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3, "model": "gpt-3.5-turbo"},
    },
    "task": {"@llm_tasks": "spacy.TextCat.v1", "labels": "POSITIVE,NEGATIVE"},
}


@registry.llm_tasks("spacy.Count.v1")
class _CountTask:
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield f"Count the number of characters in this string: '{doc.text}'."

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        return docs


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["api"] = "NoOp"
    cfg["backend"]["config"] = {"model": "NoOp"}
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_model_error_handling():
    """Test error handling for wrong model."""
    nlp = spacy.blank("en")
    with pytest.raises(ValueError) as err:
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "backend": {"config": {"model": "x-gpt-3.5-turbo"}},
            },
        )
    assert "The specified model 'x-gpt-3.5-turbo' is not available." in str(err.value)


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_doc_length_error_handling():
    """Test error handling for excessive doc length."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            # Not using the NoOp task is necessary here, as the NoOp task sends a fixed-size prompt.
            "task": {"@llm_tasks": "spacy.Count.v1"},
            "backend": {"config": {"model": "ada"}},
        },
    )
    # Call with an overly long document to elicit error.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Request to OpenAI API failed: This model's maximum context length is 2049 tokens, however you requested "
            "2527 tokens (2511 in your prompt; 16 for the completion). Please reduce your prompt; or completion length."
        ),
    ):
        nlp("n" * 5000)


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_max_time_error_handling():
    """Test error handling for exceeding max. time."""
    nlp = spacy.blank("en")
    with pytest.raises(
        TimeoutError,
        match="Request time out. Check your network connection and the API's availability.",
    ):
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.Count.v1"},
                "backend": {
                    "config": {"model": "ada"},
                    "max_request_time": 0.001,
                    "max_tries": 1,
                    "interval": 0.001,
                },
            },
        )
