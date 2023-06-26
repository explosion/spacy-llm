import copy
import os
import time
from pathlib import Path
from typing import Dict

import pytest
import spacy
import srsly  # type: ignore[import]
from spacy.language import Language
from spacy.tokens import DocBin

from ..cache import BatchCache
from .compat import has_openai_key

_DEFAULT_CFG = {
    "backend": {"api": "NoOp", "config": {"model": "NoOp"}},
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
    "cache": {
        "batch_size": 2,
        "max_batches_in_mem": 3,
    },
}


def _init_nlp(tmp_dir: Path) -> Language:
    nlp = spacy.blank("en")
    config = copy.deepcopy(_DEFAULT_CFG)
    config["cache"]["path"] = str(tmp_dir)  # type: ignore
    nlp.add_pipe("llm", config=config)
    return nlp


@pytest.mark.parametrize("use_pipe", (False, True))
def test_caching(use_pipe: bool) -> None:
    """Test pipeline with caching.
    use_pipe (bool): Whether to use .pipe().
    """
    n = 10

    with spacy.util.make_tempdir() as tmpdir:
        nlp = _init_nlp(tmpdir)
        texts = [f"Test {i}" for i in range(n)]
        # Test writing to cache dir.
        docs = list(nlp.pipe(texts)) if use_pipe else [nlp(text) for text in texts]

        #######################################################
        # Test cache writing
        #######################################################

        index = list(srsly.read_jsonl(tmpdir / "index.jsonl"))
        index_dict: Dict[int, int] = {}
        for rec in index:
            index_dict = {**index_dict, **{int(k): int(v) for k, v in rec.items()}}
        assert len(index) == len(index_dict) == n
        cache = nlp.get_pipe("llm")._cache  # type: ignore
        assert cache._stats["hit"] == 0
        assert cache._stats["hit_contains"] == 0
        assert cache._stats["missed"] == 0
        assert cache._stats["missed_contains"] == n
        assert cache._stats["added"] == n
        assert cache._stats["persisted"] == n
        # Check whether docs are in the batch files they are supposed to be in.
        for doc in docs:
            doc_id = BatchCache._doc_id(doc)
            batch_id = index_dict[doc_id]
            batch_path = cache._batch_path(batch_id)
            batch_docs = DocBin().from_disk(batch_path).get_docs(nlp.vocab)
            doc_ids = [BatchCache._doc_id(d) for d in batch_docs]
            assert BatchCache._batch_id(doc_ids) == batch_id
            assert doc_id in doc_ids

        #######################################################
        # Test cache reading
        #######################################################

        nlp_2 = _init_nlp(tmpdir)
        [nlp_2(text) for text in texts]
        cache = nlp_2.get_pipe("llm")._cache  # type: ignore
        assert cache._stats["hit"] == n
        assert cache._stats["hit_contains"] == n
        assert cache._stats["missed"] == 0
        assert cache._stats["missed_contains"] == 0
        assert cache._stats["added"] == 0
        assert cache._stats["persisted"] == 0


@pytest.mark.skip(reason="Flaky test - needs to be updated")
def test_caching_interrupted() -> None:
    """Test pipeline with caching with simulated interruption (i. e. pipeline stops writing before entire batch is
    done).
    """
    n = 100
    texts = [f"Test {i}" for i in range(n)]

    # Collect stats for complete run with caching.
    with spacy.util.make_tempdir() as tmpdir:
        nlp = _init_nlp(tmpdir)
        start = time.time()
        [nlp(text) for text in texts]
        ref_duration = time.time() - start

    with spacy.util.make_tempdir() as tmpdir:
        nlp2 = _init_nlp(tmpdir)
        # Write half of all docs.
        start = time.time()
        for i in range(int(n / 2)):
            nlp2(texts[i])
        pass1_duration = time.time() - start
        pass1_cache = nlp2.get_pipe("llm")._cache  # type: ignore
        # Arbitrary time check to ensure that first pass through half of the doc batch takes up roughly half of the time
        # of a full pass.
        assert abs(ref_duration / 2 - pass1_duration) < ref_duration / 2 * 0.3
        assert pass1_cache._stats["hit"] == 0
        assert pass1_cache._stats["hit"] == 0
        assert pass1_cache._stats["missed"] == n / 2
        assert pass1_cache._stats["missed_contains"] == n / 2
        assert pass1_cache._stats["added"] == n / 2
        assert pass1_cache._stats["persisted"] == n / 2

        nlp3 = _init_nlp(tmpdir)
        start = time.time()
        for i in range(n):
            nlp3(texts[i])
        pass2_duration = time.time() - start
        cache = nlp3.get_pipe("llm")._cache  # type: ignore
        # Arbitrary time check to ensure second pass (leveraging caching) is at least 30% faster (re-utilizing 50% of
        # the entire doc batch, so max. theoretical speed-up is 50%).
        assert ref_duration - pass2_duration >= ref_duration * 0.3
        assert cache._stats["hit"] == n / 2
        assert cache._stats["hit_contains"] == n / 2
        assert cache._stats["missed"] == n / 2
        assert cache._stats["missed_contains"] == n / 2
        assert cache._stats["added"] == n / 2
        assert cache._stats["persisted"] == n / 2


def test_path_file_invalid():
    with spacy.util.make_tempdir() as tmpdir:
        # File path instead of directory path.
        open(tmpdir / "empty_file", "a").close()
        with pytest.raises(
            ValueError, match="Cache directory exists and is not a directory."
        ):
            config = copy.deepcopy(_DEFAULT_CFG)
            config["cache"]["path"] = str(tmpdir / "empty_file")
            spacy.blank("en").add_pipe("llm", config=config)


def test_path_dir_created():
    with spacy.util.make_tempdir() as tmpdir:
        # Non-existing cache directory should be created.
        config = copy.deepcopy(_DEFAULT_CFG)
        assert not (tmpdir / "new_dir").exists()
        config["cache"]["path"] = str(tmpdir / "new_dir")
        spacy.blank("en").add_pipe("llm", config=config)
        assert (tmpdir / "new_dir").exists()


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_caching_llm_io() -> None:
    """Test availability of LLM IO after caching."""
    with spacy.util.make_tempdir() as tmpdir:
        config = copy.deepcopy(_DEFAULT_CFG)
        config["backend"]["api"] = "OpenAI"  # type: ignore[index]
        config["backend"]["config"] = {"model": "gpt-3.5-turbo"}  # type: ignore[index]
        config["cache"]["path"] = str(tmpdir)  # type: ignore[index]
        config["cache"]["batch_size"] = 3  # type: ignore[index]
        config["save_io"] = True
        nlp = spacy.blank("en")
        nlp.add_pipe("llm", config=config)
        docs = [nlp(txt) for txt in ("What's 1+1?", "What's 2+2?", "What's 3+3?")]

        assert all([doc.user_data["llm_io"]["llm"]] for doc in docs)
        cached_file_names = [
            f
            for f in os.listdir(tmpdir)
            if os.path.isfile(tmpdir / f) and f.endswith(".spacy")
        ]
        assert len(cached_file_names) == 1
        cached_docs = list(
            DocBin().from_disk(tmpdir / cached_file_names[0]).get_docs(nlp.vocab)
        )
        assert len(cached_docs) == 3
        assert all([doc.user_data["llm_io"]["llm"]] for doc in cached_docs)
