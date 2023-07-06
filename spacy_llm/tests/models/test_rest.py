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
    "model": {
        "@llm_models": "spacy.GPT-3-5.v1",
        "config": {"temperature": 0.3},
    },
    "task": {"@llm_tasks": "spacy.TextCat.v1", "labels": "POSITIVE,NEGATIVE"},
}


@registry.llm_tasks("spacy.Count.v1")
class _CountTask:
    _PROMPT_TEMPLATE = "Count the number of characters in this string: '{text}'."

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield _CountTask._PROMPT_TEMPLATE.format(text=doc.text)

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        return docs

    @property
    def prompt_template(self) -> str:
        return _CountTask._PROMPT_TEMPLATE


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["model"] = {"@llm_models": "spacy.NoOp.v1"}
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_model_error_handling():
    """Test error handling for wrong model."""
    nlp = spacy.blank("en")
    with pytest.raises(ValueError, match="Could not find function 'spacy.gpt-3.5x.v1'"):
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "model": {"@llm_models": "spacy.gpt-3.5x.v1"},
            },
        )


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
            "model": {"config": {"model": "ada"}},
        },
    )
    # Call with an overly long document to elicit error.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Request to OpenAI API failed: This model's maximum context length is 4097 tokens. However, your messages "
            "resulted in 5018 tokens. Please reduce the length of the messages."
        ),
    ):
        nlp("n" * 10000)


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
                "model": {
                    "config": {"model": "ada"},
                    "max_request_time": 0.001,
                    "max_tries": 1,
                    "interval": 0.001,
                },
            },
        )
