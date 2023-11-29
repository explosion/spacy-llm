import numbers

import pytest
from confection import Config

from spacy_llm.tests.compat import has_openai_key
from spacy_llm.util import assemble_from_config

from .util import ShardingCountTask  # noqa: F401

_CONTEXT_LENGTH = 20
_TEXT = "Do one thing every day that scares you. The only thing we have to fear is fear itself."


@pytest.fixture
def config():
    return Config().from_str(
        f"""
            [nlp]
            lang = "en"
            pipeline = ["llm"]

            [components]

            [components.llm]
            factory = "llm"
            save_io = True

            [components.llm.task]
            @llm_tasks = "spacy.CountWithSharding.v1"

            [components.llm.model]
            @llm_models = "spacy.GPT-3-5.v3"
            context_length = {_CONTEXT_LENGTH}
        """
    )


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_count(config):
    """Tests whether task shards data as expected."""
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT)
    marker = "(and nothing else): '"
    prompts = [
        pr[pr.index(marker) + len(marker) : -1]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    responses = [int(r) for r in doc.user_data["llm_io"]["llm"]["response"]]
    assert prompts == [
        "Do one thing every day ",
        "that scares you",
        ". The only ",
        "thing we have to ",
        "fear is fear itself.",
    ]
    assert all(
        [response == len(pr.split()) for response, pr in zip(responses, prompts)]
    )
    assert sum(responses) == doc.user_data["count"]


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_lemma(config):
    context_length = 120
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {"@llm_tasks": "spacy.Lemma.v1"}
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT)
    marker = "to be lemmatized:\n'''\n"
    prompts = [
        pr[pr.index(marker) + len(marker) : -4]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    # Make sure lemmas are set (somme might not be because the LLM didn't return parsable a response).
    assert any([t.lemma != 0 for t in doc])
    assert prompts == [
        "Do one thing every day that scares you. The ",
        "only thing we have to fear is fear itself.",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 2


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_ner(config):
    context_length = 265
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {
        "@llm_tasks": "spacy.NER.v3",
        "labels": ["LOCATION"],
    }
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT + " Paris is a city.")
    marker = "Paragraph: "
    prompts = [
        pr[pr.rindex(marker) + len(marker) : pr.rindex("\nAnswer:")]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    assert len(doc.ents)
    assert prompts == [
        "Do one thing every day that scares you. The only thing ",
        "we have to fear is fear itself. Paris is a city.",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 2


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_rel(config):
    context_length = 100
    config["nlp"]["pipeline"] = ["ner", "llm"]
    config["components"]["ner"] = {"source": "en_core_web_md"}
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {
        "@llm_tasks": "spacy.REL.v1",
        "labels": "LivesIn,Visits",
    }
    config["initialize"] = {"vectors": "en_core_web_md"}
    nlp = assemble_from_config(config)

    doc = nlp("Joey rents a place in New York City, which is in North America.")
    marker = "Text:\n'''\n"
    prompts = [
        pr[pr.rindex(marker) + len(marker) : -4]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    assert len(doc.ents)
    assert hasattr(doc._, "rel") and len(doc._.rel)
    assert prompts == [
        "Joey[ENT0:PERSON] rents a place in New York City",
        "[ENT1:GPE], which is in North America[ENT2:LOC].",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 2


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_sentiment(config):
    context_length = 50
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {"@llm_tasks": "spacy.Sentiment.v1"}
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT)
    marker = "Text:\n'''\n"
    prompts = [
        pr[pr.index(marker) + len(marker) : pr.rindex("\n'''\nAnswer:")]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    assert hasattr(doc._, "sentiment") and isinstance(doc._.sentiment, numbers.Number)
    assert prompts == [
        "Do one thing every day that scares you. The ",
        "only thing we have to fear is fear itself.",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 2


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_spancat(config):
    context_length = 265
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {
        "@llm_tasks": "spacy.SpanCat.v3",
        "labels": ["LOCATION"],
    }
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT + " Paris is a city.")
    marker = "Paragraph: "
    prompts = [
        pr[pr.rindex(marker) + len(marker) : pr.rindex("\nAnswer:")]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    assert len(doc.spans.data["sc"])
    assert prompts == [
        "Do one thing every day that ",
        "scares you. The only thing we have to ",
        "fear is fear itself. Paris is a city.",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 3


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_sharding_summary(config):
    context_length = 50
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {"@llm_tasks": "spacy.Summarization.v1"}
    nlp = assemble_from_config(config)

    doc = nlp(_TEXT)
    marker = "needs to be summarized:\n'''\n"
    prompts = [
        pr[pr.rindex(marker) + len(marker) : pr.rindex("\n'''\nSummary:")]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    assert hasattr(doc._, "summary") and doc._.summary
    assert prompts == [
        "Do one thing every day that scares you. The ",
        "only thing we have to fear is fear itself.",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 2
