import pytest
from confection import Config

from spacy_llm.tests.compat import has_openai_key
from spacy_llm.util import assemble_from_config

from .util import ShardingCountTask  # noqa: F401

_CONTEXT_LENGTH = 20


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
    """Tests whether tasks shard data as expected."""
    config["components"]["llm"]["model"]["@llm_models"] = model
    nlp = assemble_from_config(config)
    doc = nlp(
        "Do one thing every day that scares you. The only thing we have to fear is fear itself."
    )

    # With a context length of 20 we expect the doc to be split into five prompts.
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
    """Tests whether tasks shard data as expected."""
    context_length = 120
    config["components"]["llm"]["model"]["@llm_models"] = model
    config["components"]["llm"]["model"]["context_length"] = context_length
    config["components"]["llm"]["task"] = {"@llm_tasks": "spacy.Lemma.v1"}

    text = (
        "Do one thing every day that scares you. The only thing we have to fear is fear itself. Do one thing every "
        "day that scares you. The only thing we have to fear is fear itself. "
    )
    nlp = assemble_from_config(config)
    doc = nlp(text)

    # With a context length of 120 we expect the doc to be split into four prompts.
    marker = "to be lemmatized:\n'''\n"
    prompts = [
        pr[pr.index(marker) + len(marker) : -4]
        for pr in doc.user_data["llm_io"]["llm"]["prompt"]
    ]
    # Make sure lemmas are set (somme might not be because the LLM didn't return parsable a response).
    assert any([t.lemma != 0 for t in doc])
    assert prompts == [
        "Do one thing every day that scares you. The ",
        "only thing we have to fear is ",
        "fear itself. Do one thing every day that scares you",
        ". The only thing we have to fear is fear itself. ",
    ]
    assert len(doc.user_data["llm_io"]["llm"]["response"]) == 4
