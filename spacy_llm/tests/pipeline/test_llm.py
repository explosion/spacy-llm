import warnings
from typing import Iterable

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from spacy_llm.tasks import NoopTask
from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import registry
from spacy_llm.compat import has_openai_key


@pytest.fixture
def nlp() -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})
    return nlp


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_llm_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_llm_pipe_empty(nlp):
    """Test call .pipe() with empty batch."""
    assert list(nlp.pipe(texts=[])) == []


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        task=NoopTask,
        backend=None,  # type: ignore
        cache={"path": None, "batch_size": 0, "max_batches_in_mem": 0},
        vocab=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        task=NoopTask,
        backend=None,  # type: ignore
        cache={"path": None, "batch_size": 0, "max_batches_in_mem": 0},
        vocab=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_type_checking_valid() -> None:
    """Test type checking for consistency between functions."""
    # Ensure default config doesn't raise warnings.
    nlp = spacy.blank("en")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_type_checking_invalid() -> None:
    """Test type checking for consistency between functions."""

    @registry.llm_tasks("IncorrectTypes.v1")
    class NoopTask_Incorrect:
        def __init__(self):
            pass

        def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[int]:
            return [0] * len(list(docs))

        def parse_responses(
            self, docs: Iterable[Doc], responses: Iterable[float]
        ) -> Iterable[Doc]:
            return docs

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning) as record:
        nlp.add_pipe(
            "llm",
            config={"task": {"@llm_tasks": "IncorrectTypes.v1"}},
        )
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "Type returned from `task.generate_prompts()` (`typing.Iterable[int]`) doesn't match type "
        "expected by `backend` (`typing.Iterable[str]`)."
    )
    assert (
        str(record[1].message)
        == "Type returned from `backend` (`typing.Iterable[str]`) doesn't match type "
        "expected by `task.parse_responses()` (`typing.Iterable[float]`)."
    )
