import pytest
from confection import Config

from spacy_llm.tests.compat import has_openai_key
from spacy_llm.util import assemble_from_config

from .util import ShardingCountTask  # noqa: F401


@pytest.fixture
def config():
    return Config().from_str(
        """
            [nlp]
            lang = "en"
            pipeline = ["llm"]

            [components]

            [components.llm]
            factory = "llm"

            [components.llm.task]
            @llm_tasks = "spacy.CountWithSharding.v1"

            [components.llm.model]
            @llm_models = "spacy.GPT-3-5.v3"
            context_length = 20
        """
    )


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("model", ("spacy.GPT-3-5.v3",))
def test_with_count_task(config, model: str):
    """Tests whether tasks shard data as expected."""
    config["components"]["llm"]["model"]["@llm_models"] = model
    nlp = assemble_from_config(config)
    # todo add tests for sharding correctness checks
    nlp("This is a first shot.")


@pytest.mark.parametrize("model", ("spacy.GPT-3.5.v3",))
@pytest.mark.parametrize("task", ("spacy.Lemma.v1",))
def test_with_all_tasks(config, model: str, task: str):
    # todo add task-specific sharding tests in task test files?
    pass
