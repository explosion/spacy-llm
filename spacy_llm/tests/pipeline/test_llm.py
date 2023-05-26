import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from thinc.api import NumpyOps, get_current_ops

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import registry
from spacy_llm.tasks import make_noop_task

from ..compat import has_openai_key
from ...cache import BatchCache


@pytest.fixture
def noop_config() -> Dict[str, Any]:
    """Returns NoOp config.
    RETURNS (Dict[str, Any]): NoOp config.
    """
    return {
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "backend": {"api": "NoOp", "config": {"model": "NoOp"}},
    }


@pytest.fixture
def nlp(noop_config) -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=noop_config)
    return nlp


def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


@pytest.mark.parametrize("n_process", [1, 2])
def test_llm_pipe(nlp: Language, n_process: int):
    """Test call .pipe()."""
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")
    docs = list(
        nlp.pipe(texts=["This is a test", "This is another test"], n_process=n_process)
    )
    assert len(docs) == 2


@pytest.mark.parametrize("n_process", [1, 2])
def test_llm_pipe_with_cache(tmp_path: Path, n_process: int):
    """Test call .pipe() with pre-cached docs"""
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")

    path = tmp_path / "cache"

    config = {
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "backend": {"api": "NoOp", "config": {"model": "NoOp"}},
        "cache": {
            "path": str(path),
            "batch_size": 1,  # Eager caching
            "max_batches_in_mem": 10,
        },
    }

    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=config)

    cached_text = "This is a cached test"

    # Run the text through, caching it.
    nlp(cached_text)

    texts = [cached_text, "This is a test", "This is another test"]

    # Run it again, along with other documents
    docs = list(nlp.pipe(texts=texts, n_process=n_process))
    assert [doc.text for doc in docs] == texts


def test_llm_pipe_empty(nlp):
    """Test call .pipe() with empty batch."""
    assert list(nlp.pipe(texts=[])) == []


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        task=make_noop_task(),
        backend=None,  # type: ignore
        cache=BatchCache(path=None, batch_size=0, max_batches_in_mem=0),
        vocab=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        task=make_noop_task(),
        backend=None,  # type: ignore
        cache=BatchCache(path=None, batch_size=0, max_batches_in_mem=0),
        vocab=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_type_checking_valid(noop_config) -> None:
    """Test type checking for consistency between functions."""
    # Ensure default config doesn't raise warnings.
    nlp = spacy.blank("en")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})


def test_type_checking_invalid(noop_config) -> None:
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
        noop_config["task"] = {"@llm_tasks": "IncorrectTypes.v1"}
        nlp.add_pipe("llm", config=noop_config)
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
