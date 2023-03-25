import pytest
import spacy
from ..pipeline import LLMWrapper  # noqa: F401


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.load("blank:en")
    nlp.add_pipe("llm")
    return nlp


def test_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


def test_call(nlp):
    """Test call with single Doc instance."""
    assert nlp("This is a test")._.llm_response


def test_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2
    assert all([doc._.llm_response for doc in docs])
