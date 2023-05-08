from typing import Dict

import pytest
import srsly
from dotenv import load_dotenv
import spacy
from spacy.tokens import DocBin

from ..cache import Cache


load_dotenv()  # take environment variables from .env.


def test_caching() -> None:
    """Test pipeline with caching."""
    n = 10

    with spacy.util.make_tempdir() as tmpdir:
        nlp = spacy.blank("en")
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "cache": {
                    "path": str(tmpdir),
                    "batch_size": 2,
                    "max_batches_in_mem": 3,
                },
            },
        )
        texts = [f"Test {i}" for i in range(n)]
        # Test writing to cache dir.
        docs = [nlp(text) for text in texts]

        #######################################################
        # Test cache writing
        #######################################################

        index = list(srsly.read_jsonl(tmpdir / "index.jsonl"))
        index_dict: Dict[int, int] = {}
        for rec in index:
            index_dict = {**index_dict, **{int(k): int(v) for k, v in rec.items()}}
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
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "cache": {
                    "path": str(tmpdir),
                    "batch_size": 2,
                    "max_batches_in_mem": 3,
                },
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
                        "max_batches_in_mem": 3,
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
                    "max_batches_in_mem": 3,
                },
            },
        )
        assert (tmpdir / "new_dir").exists()
