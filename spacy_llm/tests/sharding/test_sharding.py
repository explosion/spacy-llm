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
def test_with_count_task(config, model: str):
    """Tests whether tasks shard data as expected."""
    config["components"]["llm"]["model"]["@llm_models"] = model
    nlp = assemble_from_config(config)
    doc = nlp(
        "Do one thing every day that scares you. The only thing we have to fear is fear itself."
    )

    # With a context length of 20 we expect the doc to be split into five prompts.
    prompts = [
        pr.replace('"', "").strip()
        for pr in doc.user_data["llm_io"]["llm"]["prompt"][1:-1].split('",')
    ]
    prompt_texts = [pr[65:].replace("'", "").strip() for pr in prompts]
    responses = [
        int(r.replace("'", ""))
        for r in doc.user_data["llm_io"]["llm"]["response"][1:-1].split("',")
    ]
    assert prompt_texts == [
        "Do one thing every day",
        "that scares you",
        ". The only",
        "thing we have to",
        "fear is fear itself.",
    ]
    assert all(
        [response == len(pr.split()) for response, pr in zip(responses, prompt_texts)]
    )
