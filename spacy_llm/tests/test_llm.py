import warnings
from typing import Callable, Iterable, Tuple

import pytest
import spacy
from dotenv import load_dotenv
from spacy.tokens import Doc

from ..pipeline import LLMWrapper

load_dotenv()  # take environment variables from .env.


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm")
    return nlp


def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


def test_llm_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        backend=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        backend=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


def test_type_checking() -> None:
    """Tests type checking for consistency between functions."""

    @spacy.registry.llm_tasks("spacy.TestIncorrect.v1")
    def noop_task_incorrect() -> Tuple[
        Callable[[Iterable[Doc]], Iterable[int]],
        Callable[[Iterable[Doc], Iterable[int]], Iterable[Doc]],
    ]:
        def template(docs: Iterable[Doc]) -> Iterable[int]:
            return [0] * len(list(docs))

        def parse(
            docs: Iterable[Doc], prompt_responses: Iterable[int]
        ) -> Iterable[Doc]:
            return docs

        return template, parse

    # Ensure default config doesn't raise warnings.
    nlp = spacy.blank("en")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlp.add_pipe("llm")

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning) as record:
        nlp.add_pipe(
            "llm",
            config={"task": {"@llm_tasks": "spacy.TestIncorrect.v1"}},
        )
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "Type returned from `task[0]` (`typing.Iterable[int]`) doesn't match type "
        "expected by `backend` (`typing.Iterable[str]`)."
    )
    assert (
        str(record[1].message)
        == "Type returned from `backend` (`typing.Iterable[str]`) doesn't match type "
        "expected by `parse` (`typing.Iterable[int]`)."
    )
