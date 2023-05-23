from pathlib import Path

import pytest

from .. import ner_dolly, textcat_openai, ner_langchain_openai, ner_minichain_openai

from thinc.compat import has_torch_cuda_gpu

_USAGE_EXAMPLE_PATH = Path(__file__).parent.parent


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_ner_dolly(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "ner_dolly"
    ner_dolly.run_pipeline(
        "text",
        path / config_name,
        None if config_name == "zeroshot.cfg" else path / "examples.yml",
        False,
    )


@pytest.mark.external
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_textcat_openai(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "textcat_openai"
    textcat_openai.run_pipeline(
        "text",
        path / config_name,
        None if config_name == "zeroshot.cfg" else path / "examples.jsonl",
        False,
    )


@pytest.mark.external
def test_ner_langchain_openai():
    """Test NER LangChain OpenAI usage example."""
    ner_langchain_openai.run_pipeline(
        "text", _USAGE_EXAMPLE_PATH / "ner_langchain_openai" / "ner.cfg", False
    )


@pytest.mark.external
def test_ner_minichain_openai():
    """Test NER LangChain OpenAI usage example."""
    ner_minichain_openai.run_pipeline(
        "text", _USAGE_EXAMPLE_PATH / "ner_minichain_openai" / "ner.cfg", False
    )
