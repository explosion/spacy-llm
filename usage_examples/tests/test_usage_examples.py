from pathlib import Path

import pytest
from thinc.compat import has_torch_cuda_gpu

from spacy_llm import cache  # noqa: F401

from .. import multitask_openai, ner_dolly, ner_langchain_openai, ner_v3_openai
from .. import rel_openai, textcat_openai

_USAGE_EXAMPLE_PATH = Path(__file__).parent.parent


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize("config_name", ("fewshot_v2.cfg", "zeroshot_v2.cfg"))
def test_ner_dolly(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "ner_dolly"
    ner_dolly.run_pipeline(
        text="text",
        config_path=path / config_name,
        examples_path=None if "zeroshot" in config_name else path / "examples_v2.yml",
        verbose=False,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_ner_v3_dolly(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "ner_dolly"
    ner_dolly.run_pipeline(
        text="text",
        config_path=path / config_name,
        examples_path=None if "zeroshot" in config_name else path / "examples.yml",
        verbose=False,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_textcat_dolly(config_name: str):
    """Test Textcat Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "textcat_dolly"
    textcat_openai.run_pipeline(
        text="text",
        config_path=path / config_name,
        examples_path=None
        if config_name == "zeroshot.cfg"
        else path / "examples.jsonl",
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_textcat_openai(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "textcat_openai"
    textcat_openai.run_pipeline(
        text="text",
        config_path=path / config_name,
        examples_path=None
        if config_name == "zeroshot.cfg"
        else path / "examples.jsonl",
        verbose=False,
    )


@pytest.mark.external
def test_ner_v3_openai():
    """Test NER v3 OpenAI usage example."""
    path = _USAGE_EXAMPLE_PATH / "ner_v3_openai"
    ner_v3_openai.run_pipeline(
        text="text",
        config_path=path / "fewshot.cfg",
        examples_path=path / "examples.json",
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ner_langchain_openai():
    """Test NER LangChain OpenAI usage example."""
    ner_langchain_openai.run_pipeline(
        "text", _USAGE_EXAMPLE_PATH / "ner_langchain_openai" / "ner.cfg", False
    )


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("config_name", ("fewshot.cfg", "zeroshot.cfg"))
def test_multitask_openai(config_name: str):
    """Test multitask OpenAI example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "multitask_openai"
    multitask_openai.run_pipeline(
        text="text",
        config_path=path / config_name,
        examples_path=None if config_name == "zeroshot.cfg" else path / "examples.yml",
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.parametrize("config_name", ("zeroshot.cfg", "fewshot.cfg"))
def test_rel_openai(config_name: str):
    """Test REL OpenAI usage example.
    config_name (str): Name of config file to use.
    """
    path = _USAGE_EXAMPLE_PATH / "rel_openai"
    rel_openai.run_pipeline(
        text="Sara lives in Lisbon.",
        config_path=path / config_name,
        examples_path=None
        if config_name == "zeroshot.cfg"
        else path / "examples.jsonl",
        verbose=False,
    )
