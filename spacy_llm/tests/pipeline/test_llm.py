import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pytest
import spacy
import srsly
from confection import Config
from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import make_tempdir
from thinc.api import NumpyOps, get_current_ops

import spacy_llm
from spacy_llm.models.rest.noop import _NOOP_RESPONSE
from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import registry
from spacy_llm.tasks import _LATEST_TASKS, make_noop_task
from spacy_llm.tasks.noop import _NOOP_PROMPT, ShardingNoopTask

from ...cache import BatchCache
from ...registry.reader import fewshot_reader
from ...util import assemble_from_config
from ..compat import has_openai_key
from ..tasks.test_entity_linker import build_el_pipeline


@pytest.fixture
def noop_config() -> Dict[str, Any]:
    """Returns NoOp config.
    RETURNS (Dict[str, Any]): NoOp config.
    """
    return {
        "save_io": True,
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "model": {"@llm_models": "spacy.NoOp.v1"},
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
@pytest.mark.parametrize("shard", [True, False])
def test_llm_pipe(noop_config: Dict[str, Any], n_process: int, shard: bool):
    """Test call .pipe()."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={**noop_config, **{"task": {"@llm_tasks": "spacy.NoOpNoShards.v1"}}}
        if not shard
        else noop_config,
    )
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")

    docs = list(
        nlp.pipe(texts=["This is a test", "This is another test"], n_process=n_process)
    )
    assert len(docs) == 2

    for doc in docs:
        llm_io = doc.user_data["llm_io"]
        assert llm_io["llm"]["prompt"] == ([_NOOP_PROMPT] if shard else _NOOP_PROMPT)
        assert llm_io["llm"]["response"] == (
            [_NOOP_RESPONSE] if shard else _NOOP_RESPONSE
        )


@pytest.mark.parametrize("n_process", [2])
def test_llm_pipe_with_cache(tmp_path: Path, n_process: int):
    """Test call .pipe() with pre-cached docs"""
    ops = get_current_ops()
    if not isinstance(ops, NumpyOps) and n_process != 1:
        pytest.skip("Only test multiple processes on CPU")

    path = tmp_path / "cache"

    config = {
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "model": {"@llm_models": "spacy.NoOp.v1"},
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

    egs = [(text, i) for i, text in enumerate(texts)]
    egs_processed = list(nlp.pipe(egs, as_tuples=True, n_process=n_process))
    assert [doc.text for doc, _ in egs_processed] == texts
    assert [eg for _, eg in egs_processed] == list(range(len(texts)))


def test_llm_pipe_empty(nlp):
    """Test call .pipe() with empty batch."""
    assert list(nlp.pipe(texts=[])) == []


def test_llm_serialize_bytes():
    with pytest.warns(UserWarning, match="Task supports sharding"):
        llm = LLMWrapper(
            task=make_noop_task(),
            save_io=False,
            model=None,  # type: ignore
            cache=BatchCache(path=None, batch_size=0, max_batches_in_mem=0),
            vocab=None,  # type: ignore
        )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    with pytest.warns(UserWarning, match="Task supports sharding"):
        llm = LLMWrapper(
            task=make_noop_task(),
            save_io=False,
            model=None,  # type: ignore
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

        def generate_prompts(
            self, docs: Iterable[Doc], context_length: Optional[int] = None
        ) -> Iterable[Tuple[Iterable[int], Iterable[Doc]]]:
            for doc in docs:
                yield [0], [doc]

        def parse_responses(
            self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[float]]
        ) -> Iterable[Doc]:
            return list(shards)[0]

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning) as record:
        noop_config["task"] = {"@llm_tasks": "IncorrectTypes.v1"}
        nlp.add_pipe("llm", config=noop_config)
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "First type in value returned from `task.generate_prompts()` (`typing.Iterable[int]`) doesn't match type "
        "expected by `model` (`typing.Iterable[str]`)."
    )
    assert (
        str(record[1].message)
        == "Type returned from `model` (`typing.Iterable[typing.Iterable[str]]`) doesn't match type expected by "
        "`task.parse_responses()` (`typing.Iterable[typing.Iterable[float]]`)."
    )

    # Run with disabled type consistency validation.
    nlp = spacy.blank("en")
    noop_config["validate_types"] = False
    nlp.add_pipe("llm", config=noop_config)


@pytest.mark.parametrize("use_pipe", [True, False])
def test_llm_logs_at_debug_level(
    nlp: Language, use_pipe: bool, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO):
        if use_pipe:
            doc = next(nlp.pipe(["This is a test"]))
        else:
            doc = nlp("This is a test")

    assert "spacy_llm" not in caplog.text
    assert doc.text not in caplog.text

    with caplog.at_level(logging.DEBUG):
        if use_pipe:
            doc = next(nlp.pipe(["This is a test"]))
        else:
            doc = nlp("This is a test")

    assert "spacy_llm" in caplog.text
    assert doc.text in caplog.text

    assert f"Generated prompt for doc: {doc.text}" in caplog.text
    assert "Don't do anything" in caplog.text
    assert f"LLM response for doc: {doc.text}" in caplog.text


def test_llm_logs_default_null_handler(nlp: Language, capsys: pytest.CaptureFixture):
    nlp("This is a test")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    # Add a basic Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    spacy_llm.logger.addHandler(stream_handler)
    spacy_llm.logger.setLevel(logging.DEBUG)

    doc = nlp("This is a test")
    captured = capsys.readouterr()
    assert f"Generated prompt for doc: {doc.text}" in captured.out
    assert "Don't do anything" in captured.out
    assert f"LLM response for doc: {doc.text}" in captured.out

    # Remove the Stream Handler from the spacy_llm logger
    spacy_llm.logger.removeHandler(stream_handler)

    doc = nlp("This is a test with no handler")
    captured = capsys.readouterr()
    assert f"Generated prompt for doc: {doc.text}" not in captured.out
    assert "Don't do anything" not in captured.out
    assert f"LLM response for doc: {doc.text}" not in captured.out


def test_fewshot_reader_file_format_handling():
    """Test if fewshot reader copes with file formats as expected."""
    example = [
        {
            "text": "Circe lived on Aeaea.",
            "entities": {"PER": ["Circe"], "LOC": ["Aeaea"]},
        }
    ]
    with make_tempdir() as tmpdir:
        srsly.write_yaml(tmpdir / "example.yml", example)
        srsly.write_yaml(tmpdir / "example.json", example)
        srsly.write_yaml(tmpdir / "example.foo", example)

        fewshot_reader(tmpdir / "example.yml")
        fewshot_reader(tmpdir / "example.json")
        fewshot_reader(tmpdir / "example.foo")


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_pipe_labels():
    """Test pipe labels with serde."""

    cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.TextCat.v2"
    labels = ["COMPLIMENT", "INSULT"]

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v2"
    """

    config = Config().from_str(cfg_string)
    nlp = assemble_from_config(config)

    with spacy.util.make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir / "tst.nlp")
        nlp = spacy.load(tmpdir / "tst.nlp")
        assert nlp.pipe_labels["llm"] == ["COMPLIMENT", "INSULT"]


def test_llm_task_factories():
    """Test whether llm_TASK factories run successfully."""
    for task_handle in _LATEST_TASKS:
        # Separate test for EntityLinker in test_llm_task_factories_el().
        if "EntityLinker" in task_handle:
            continue

        cfg_string = f"""
        [nlp]
        lang = "en"
        pipeline = ["llm"]

        [components]

        [components.llm]
        factory = "llm_{task_handle.split('.')[1].lower()}"

        [components.llm.model]
        @llm_models = "test.NoOpModel.v1"
        """
        config = Config().from_str(cfg_string)

        # Translation task is expected to require a target language.
        if "Translation" in task_handle:
            config["components"]["llm"]["task"] = {"target_lang": "Spanish"}

        with pytest.warns(UserWarning, match="Task supports sharding"):
            assemble_from_config(config)


def test_llm_task_factories_el(tmp_path):
    """Test whether llm_entity_linking factory runs successfully. It's necessary to do this separately, as the EL task
    requires a non-defaultable extra config setup and knowledge base."""
    cfg = """
    [paths]
    el_nlp = null
    el_kb = null
    el_desc = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm_entitylinker"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"

    [components.llm.task]
    @llm_tasks = "spacy.EntityLinker.v1"

    [initialize]
    [initialize.components]
    [initialize.components.llm]

    [initialize.components.llm.candidate_selector]
    @llm_misc = "spacy.CandidateSelector.v1"

    [initialize.components.llm.candidate_selector.kb_loader]
    @llm_misc = "spacy.KBObjectLoader.v1"
    path = ${paths.el_kb}
    nlp_path = ${paths.el_nlp}
    desc_path = ${paths.el_desc}
    """
    config = Config().from_str(
        cfg,
        overrides={
            "paths.el_nlp": str(tmp_path),
            "paths.el_kb": str(tmp_path / "entity_linker" / "kb"),
            "paths.el_desc": str(tmp_path / "desc.csv"),
        },
    )
    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    with pytest.warns(UserWarning, match="Task supports sharding"):
        assemble_from_config(config)


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_llm_task_factories_ner():
    """Test whether llm_ner behaves as expected."""
    cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm_ner"

    [components.llm.task]
    labels = PER,ORG,LOC

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """
    config = Config().from_str(cfg_string)
    nlp = assemble_from_config(config)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)

    assert len(doc.ents) > 0
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.parametrize("shard", [True, False])
def test_llm_custom_data(noop_config: Dict[str, Any], shard: bool):
    """Test whether custom doc data is preserved."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={**noop_config, **{"task": {"@llm_tasks": "spacy.NoOpNoShards.v1"}}}
        if not shard
        else noop_config,
    )

    doc = nlp.make_doc("This is a test")
    if not Doc.has_extension("test"):
        Doc.set_extension("test", default=None)
    doc._.test = "Test"
    doc.user_data["test"] = "Test"

    doc = nlp(doc)
    assert doc._.test == "Test"
    assert doc.user_data["test"] == "Test"


def test_llm_custom_data_overwrite(noop_config: Dict[str, Any]):
    """Test whether custom doc data is overwritten as expected."""

    class NoopTaskWithCustomData(ShardingNoopTask):
        def parse_responses(
            self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
        ) -> Iterable[Doc]:
            docs = super().parse_responses(shards, responses)
            for doc in docs:
                doc._.test = "Test 2"
                doc.user_data["test"] = "Test 2"
            return docs

    @registry.llm_tasks("spacy.NoOpCustomData.v1")
    def make_noopnoshards_task():
        return NoopTaskWithCustomData()

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={**noop_config, **{"task": {"@llm_tasks": "spacy.NoOpCustomData.v1"}}},
    )
    doc = nlp.make_doc("This is a test")
    for extension in ("test", "test_nooverwrite"):
        if not Doc.has_extension(extension):
            Doc.set_extension(extension, default=None)
    doc._.test = "Test"
    doc._.test_nooverwrite = "Test"
    doc.user_data["test"] = "Test"
    doc.user_data["test_nooverwrite"] = "Test"

    doc = nlp(doc)
    assert doc._.test == "Test 2"
    assert doc.user_data["test"] == "Test 2"
    assert doc._.test_nooverwrite == "Test"
    assert doc.user_data["test_nooverwrite"] == "Test"
