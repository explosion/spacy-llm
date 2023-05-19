# mypy: ignore-errors
import copy
import re
from typing import Iterable

import pytest
import spacy
from spacy.tokens import Doc

from ..compat import has_openai_key
from ...registry import registry

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.REST.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3, "model": "gpt-3.5-turbo"},
    },
    "task": {"@llm_tasks": "spacy.TextCat.v1", "labels": "POSITIVE,NEGATIVE"},
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["api"] = "NoOp"
    cfg["backend"]["config"] = {"model": "NoOp"}
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")


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


@pytest.mark.external
def test_doc_length_error_handling():
    """Test error handling for wrong URL."""
    nlp = spacy.blank("en")

    @registry.llm_tasks("spacy.Count.v1")
    class CountTask:
        def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
            for doc in docs:
                yield f"Count the number of characters in this string: '{doc.text}'."

        def parse_responses(
            self, docs: Iterable[Doc], responses: Iterable[str]
        ) -> Iterable[Doc]:
            return docs

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


@pytest.mark.parametrize("model", ("gpt-3.5-turbo", "text-davinci-002"))
@pytest.mark.external
def test_openai(model: str):
    """Test OpenAI call to /chat/completions and /completions backend.
    model (str): Model to use.
    """
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["config"]["model"] = model
    cfg["backend"]["config"]["url"] = (
        "https://api.openai.com/v1/chat/completions"
        if model == "gpt-3.5-turbo"
        else "https://api.openai.com/v1/completions"
    )
    nlp.add_pipe(
        "llm",
        config=cfg,
    )
    nlp("test")
    docs = list(nlp.pipe(["test 1", "test 2"]))
    assert len(docs) == 2
    assert docs[0].text == "test 1"
    assert docs[1].text == "test 2"


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_model_backend_compatibility():
    """Tests whether incompatible model and backend are detected as expected."""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["backend"]["config"]["model"] = "gpt-4"
    cfg["backend"]["config"]["url"] = "https://api.openai.com/v1/completions"
    with pytest.warns(
        UserWarning,
        match="Configured endpoint https://api.openai.com/v1/completions diverges from expected endpoint "
        "https://api.openai.com/v1/chat/completions for selected model 'gpt-4'. Please ensure that this endpoint "
        "supports your model.",
    ):
        nlp.add_pipe(
            "llm",
            config=cfg,
        )
