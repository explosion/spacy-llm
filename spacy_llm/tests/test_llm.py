import pytest
import spacy
from ..pipeline import LLMWrapper  # noqa: F401
from ..api import MiniChain


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm")
    return nlp


def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


def test_llm_call(nlp):
    """Test call with single Doc instance."""
    nlp("This is a test")


def test_llm_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        template=None,  # type: ignore
        api=lambda: MiniChain("OpenAI", prompt=None, backend_config={}),  # type: ignore
        parse=None,  # type: ignore
    )
    assert llm._api._backend_id == "OpenAI"

    new_llm = LLMWrapper(
        template=None,  # type: ignore
        api=lambda: MiniChain("HuggingFace", prompt=None, backend_config={}),  # type: ignore
        parse=None,  # type: ignore
    ).from_bytes(llm.to_bytes())
    assert new_llm._api._backend_id == llm._api._backend_id == "OpenAI"


def test_llm_serialize_disk():
    llm = LLMWrapper(
        template=None,  # type: ignore
        api=lambda: MiniChain("OpenAI", prompt=None, backend_config={}),  # type: ignore
        parse=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        new_llm = LLMWrapper(
            template=None,  # type: ignore
            api=lambda: MiniChain("HuggingFace", prompt=None, backend_config={}),  # type: ignore
            parse=None,  # type: ignore
        ).from_disk(tmp_dir / "llm")
    assert new_llm._api._backend_id == llm._api._backend_id == "OpenAI"


def test_llm_langchain():
    """Test configuration with LangChain."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "api": {
                "@llm": "spacy.api.LangChain.v1",
                "backend": "openai",
                "backend_config": {"temperature": 0.3},
                "prompt": {"@llm": "spacy.prompt.LangChainSimple.v1"},
            }
        },
    )
