import pytest
import spacy
from ..pipeline import LLMWrapper  # noqa: F401


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.load("blank:en")
    nlp.add_pipe("llm")
    return nlp


def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


def test_llm_call(nlp):
    """Test call with single Doc instance."""
    assert nlp("This is a test")._.llm_response


def test_llm_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2
    assert all([doc._.llm_response for doc in docs])


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        backend="OpenAI", response_field="llm_wrapper", prompt=None, batch_prompt=None  # type: ignore
    )
    assert llm._backend_id == "OpenAI"
    assert llm._response_field == "llm_wrapper"
    bytes_data = llm.to_bytes()
    new_llm = LLMWrapper(
        backend="Google", response_field=None, prompt=None, batch_prompt=None  # type: ignore
    ).from_bytes(bytes_data)
    assert new_llm._backend_id == llm._backend_id
    assert new_llm._response_field == llm._response_field


def test_llm_serialize_disk():
    llm = LLMWrapper(
        backend="OpenAI", response_field="llm_wrapper", prompt=None, batch_prompt=None  # type: ignore
    )
    assert llm._backend_id == "OpenAI"
    assert llm._response_field == "llm_wrapper"
    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        new_llm = LLMWrapper(
            backend="Google", response_field=None, prompt=None, batch_prompt=None  # type: ignore
        ).from_disk(tmp_dir / "llm")
    assert new_llm._backend_id == llm._backend_id
    assert new_llm._response_field == llm._response_field
