from typing import Dict

import pytest
import srsly
from dotenv import load_dotenv
import spacy
from spacy.tokens import DocBin
import copy

from ..cache import Cache


load_dotenv()  # take environment variables from .env.

DEFAULT_CONFIG = {
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
    "cache": {
        "batch_size": 2,
        "max_batches_in_mem": 3,
    },
}


def test_caching() -> None:
    """Test pipeline with caching."""
    n = 10

    with spacy.util.make_tempdir() as tmpdir:
        nlp = spacy.blank("en")
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["cache"]["path"] = str(tmpdir)
        nlp.add_pipe("llm", config=config)
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
            doc_id = Cache._doc_id(doc)
            batch_id = index_dict[doc_id]
            batch_path = cache._batch_path(batch_id)
            batch_docs = DocBin().from_disk(batch_path).get_docs(nlp.vocab)
            doc_ids = [Cache._doc_id(d) for d in batch_docs]
            assert Cache._batch_id(doc_ids) == batch_id
            assert doc_id in doc_ids

        #######################################################
        # Test cache reading
        #######################################################

        nlp_2 = spacy.blank("en")
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["cache"]["path"] = str(tmpdir)
        nlp_2.add_pipe("llm", config=config)
        [nlp_2(text) for text in texts]
        cache = nlp_2.get_pipe("llm")._cache
        assert cache._stats["hit"] == n
        assert cache._stats["missed"] == 0
        assert cache._stats["added"] == 0
        assert cache._stats["persisted"] == 0


def test_path_file_invalid():
    with spacy.util.make_tempdir() as tmpdir:
        # File path instead of directory path.
        open(tmpdir / "empty_file", "a").close()
        with pytest.raises(
            ValueError, match="Cache directory exists and is not a directory."
        ):
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["cache"]["path"] = str(tmpdir / "empty_file")
            spacy.blank("en").add_pipe("llm", config=config)


def test_path_dir_created():
    with spacy.util.make_tempdir() as tmpdir:
        # Non-existing cache directory should be created.
        config = copy.deepcopy(DEFAULT_CONFIG)
        assert not (tmpdir / "new_dir").exists()
        config["cache"]["path"] = str(tmpdir / "new_dir")
        spacy.blank("en").add_pipe("llm", config=config)
        assert (tmpdir / "new_dir").exists()
