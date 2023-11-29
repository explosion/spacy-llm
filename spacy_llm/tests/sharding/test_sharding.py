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
@pytest.mark.parametrize("model", ("spacy.GPT-3-5.v3",))
def test_sharding_count(config, model: str):
    """Tests whether task shards data as expected."""
    config["components"]["llm"]["model"]["@llm_models"] = model
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
@pytest.mark.parametrize("model", ("spacy.GPT-3-5.v3",))
def test_sharding_lemma(config, model: str):
    """Tests whether task shards data as expected."""
    context_length = 120
    config["components"]["llm"]["model"]["@llm_models"] = model
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
@pytest.mark.parametrize("model", ("spacy.GPT-3-5.v3",))
def test_sharding_ner(config, model: str):
    """Tests whether task shards data as expected."""
    context_length = 265
    config["components"]["llm"]["model"]["@llm_models"] = model
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
