import warnings
from typing import Any, Callable, Dict, Iterable, Tuple

import pytest
import srsly
from dotenv import load_dotenv
import spacy
from spacy.tokens import Doc, DocBin

from ..pipeline import LLMWrapper
from ..cache import Cache


load_dotenv()  # take environment variables from .env.


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})
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
        cache={"path": None, "batch_size": 0, "max_n_batches": 0},
        vocab=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        backend=None,  # type: ignore
        cache={"path": None, "batch_size": 0, "max_n_batches": 0},
        vocab=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


@pytest.mark.parametrize(
    "config",
    (
        {
            "query": "spacy.RunMiniChain.v1",
            "backend": "spacy.MiniChain.v1",
            "api": "OpenAI",
            "config": {},
        },
        {
            "query": "spacy.CallLangChain.v1",
            "backend": "spacy.LangChain.v1",
            "api": "openai",
            "config": {"temperature": 0.3},
        },
    ),
)
def test_integrations(config: Dict[str, Any]):
    """Test simple runs with all supported integrations."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {"@llm_tasks": "spacy.NoOp.v1"},
            "backend": {
                "api": config["api"],
                "@llm_backends": config["backend"],
                "config": {},
                "query": {"@llm_queries": config["query"]},
            },
        },
    )
    nlp("This is a test.")


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
        nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})

    nlp = spacy.blank("en", config={"task": {"@llm_tasks": "spacy.NoOp.v1"}})
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


def test_caching() -> None:
    """Test pipeline with caching."""
    n = 10

    with spacy.util.make_tempdir() as tmpdir:
        nlp = spacy.blank("en")
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "cache": {"path": str(tmpdir), "batch_size": 2, "max_n_batches": 3},
            },
        )
        texts = [f"Test {i}" for i in range(n)]
        # Test writing to cache dir.
        docs = [nlp(text) for text in texts]

        #######################################################
        # Test cache writing
        #######################################################

        index = list(srsly.read_jsonl(tmpdir / "index.jsonl"))
        index_dict: Dict[str, str] = {}
        for rec in index:
            index_dict |= rec
        assert len(index) == len(index_dict) == n
        cache = nlp.get_pipe("llm")._cache
        assert cache._stats["hit"] == 0
        assert cache._stats["missed"] == n
        assert cache._stats["added"] == n
        assert cache._stats["persisted"] == n
        # Check whether docs are in the batch files they are supposed to be in.
        for doc in docs:
            doc_id = Cache._id([doc])
            batch_id = index_dict[doc_id]
            batch_docs = list(
                DocBin().from_disk(tmpdir / f"{batch_id}.spacy").get_docs(nlp.vocab)
            )
            assert Cache._id(batch_docs) == batch_id
            assert doc_id in {Cache._id([batch_doc]) for batch_doc in batch_docs}

        #######################################################
        # Test cache reading
        #######################################################

        nlp = spacy.blank("en")
        nlp.add_pipe(
            "llm",
            config={
                "cache": {"path": str(tmpdir), "batch_size": 2, "max_n_batches": 3},
            },
        )
        [nlp(text) for text in texts]
        cache = nlp.get_pipe("llm")._cache
        assert cache._stats["hit"] == n
        assert cache._stats["missed"] == 0
        assert cache._stats["added"] == 0
        assert cache._stats["persisted"] == 0

        #######################################################
        # Test path handling
        #######################################################

        # File path instead of directory path.
        open(tmpdir / "empty_file", "a").close()
        with pytest.raises(
            ValueError, match="Cache directory exists and is not a directory."
        ):
            spacy.blank("en").add_pipe(
                "llm",
                config={
                    "task": {"@llm_tasks": "spacy.NoOp.v1"},
                    "cache": {
                        "path": str(tmpdir / "empty_file"),
                        "batch_size": 2,
                        "max_n_batches": 3,
                    },
                },
            )

        # Non-existing cache directory should be created.
        spacy.blank("en").add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "cache": {
                    "path": str(tmpdir / "new_dir"),
                    "batch_size": 2,
                    "max_n_batches": 3,
                },
            },
        )
        assert (tmpdir / "new_dir").exists()
