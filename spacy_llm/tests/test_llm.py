import pytest
import spacy
from ..pipeline import LLMWrapper  # noqa: F401
from ..api import MiniChain


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
        response_field="llm_wrapper",
        template=None,  # type: ignore
        api=lambda: MiniChain("OpenAI", prompt=None, core_config=None),  # type: ignore
        parse=None,  # type: ignore
    )
    assert llm._api._backend_id == "OpenAI"
    assert llm._response_field == "llm_wrapper"

    new_llm = LLMWrapper(
        response_field=None,
        template=None,  # type: ignore
        api=lambda: MiniChain("HuggingFace", prompt=None, core_config=None),  # type: ignore
        parse=None,  # type: ignore
    ).from_bytes(llm.to_bytes())
    assert new_llm._api._backend_id == llm._api._backend_id == "OpenAI"
    assert new_llm._response_field == llm._response_field


def test_llm_serialize_disk():
    llm = LLMWrapper(
        response_field="llm_wrapper",
        template=None,  # type: ignore
        api=lambda: MiniChain("OpenAI", prompt=None, core_config=None),  # type: ignore
        parse=None,  # type: ignore
    )
    assert llm._api._backend_id == "OpenAI"
    assert llm._response_field == "llm_wrapper"

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        new_llm = LLMWrapper(
            response_field=None,
            template=None,  # type: ignore
            api=lambda: MiniChain("HuggingFace", prompt=None, core_config=None),  # type: ignore
            parse=None,  # type: ignore
        ).from_disk(tmp_dir / "llm")
    assert new_llm._api._backend_id == llm._api._backend_id == "OpenAI"
    assert new_llm._response_field == llm._response_field


def test_llm_langchain():
    """Test configuration with LangChain."""
    nlp = spacy.load("blank:en")
    nlp.add_pipe(
        "llm",
        config={
            "api": {
                "@llm": "spacy.api.LangChain.v1",
                "backend": "openai",
                "core_config": {"temperature": 0.3},
                "prompt": {"@llm": "spacy.prompt.LangChainSimple.v1"},
            }
        },
    )
