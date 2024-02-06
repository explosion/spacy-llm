import os
from typing import List

import pytest
import spacy

from spacy_llm.compat import has_langchain
from spacy_llm.models.langchain import LangChain
from spacy_llm.tests.compat import has_azure_openai_key

PIPE_CFG = {
    "model": {
        "@llm_models": "langchain.OpenAIChat.v1",
        "name": "gpt-3.5-turbo",
        "config": {"temperature": 0.3},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


def langchain_model_reg_handles() -> List[str]:
    """Returns a list of all LangChain model reg handles."""
    return [
        f"langchain.{cls.__name__}.v1"
        for class_id, cls in LangChain.get_type_to_cls_dict().items()
    ]


@pytest.mark.external
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


@pytest.mark.external
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
@pytest.mark.skipif(
    has_azure_openai_key is False, reason="Azure OpenAI key not available"
)
def test_initialization_azure_openai():
    """Test initialization and simple run with Azure OpenAI."""
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_KEY"]
    _pipe_cfg = {
        "model": {
            "@llm_models": "langchain.AzureOpenAI.v1",
            "name": "gpt-35-turbo",
            "config": {
                "deployment_name": "gpt-35-turbo",
                "openai_api_version": "2023-05-15",
                "openai_api_base": "https://explosion.openai.azure.com/",
            },
        },
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
    }

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp.add_pipe("llm", config=_pipe_cfg)
    nlp("This is a test.")
