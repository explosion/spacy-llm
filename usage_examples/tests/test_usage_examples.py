from pathlib import Path

import pytest
from thinc.compat import has_torch_cuda_gpu

from ..multitask_openai import run_multitask_openai_pipeline
from ..ner_dolly import run_ner_dolly_pipeline
from ..textcat_openai import run_textcat_openai_pipeline
from ..rel_openai import run_rel_openai_pipeline

_USAGE_EXAMPLE_PATH = Path(__file__).parent.parent


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize(
    "config_name", ("dolly_ner_fewshot.cfg", "dolly_ner_zeroshot.cfg")
)
def test_ner_dolly(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    run_ner_dolly_pipeline.run_pipeline(
        text="text",
        config_path=_USAGE_EXAMPLE_PATH / "ner_dolly" / config_name,
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.parametrize(
    "config_name", ("openai_textcat_fewshot.cfg", "openai_textcat_zeroshot.cfg")
)
def test_textcat_openai(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    run_textcat_openai_pipeline.run_pipeline(
        text="text",
        config_path=_USAGE_EXAMPLE_PATH / "textcat_openai" / config_name,
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.parametrize(
    "config_name", ("openai_multitask_fewshot.cfg", "openai_multitask_zeroshot.cfg")
)
def test_multitask_openai(config_name: str):
    """Test multi-task OpenAI usage example.
    config_name (str): Name of config file to use.
    """
    run_multitask_openai_pipeline.run_pipeline(
        text="text",
        config_path=_USAGE_EXAMPLE_PATH / "multitask_openai" / config_name,
        verbose=False,
    )


@pytest.mark.external
@pytest.mark.parametrize(
    "config_name", ("openai_rel_fewshot.cfg", "openai_rel_zeroshot.cfg")
)
def test_re_openai(config_name: str):
    """Test REL OpenAI usage example.
    config_name (str): Name of config file to use.
    """
    run_rel_openai_pipeline.run_pipeline(
        text="text",
        config_path=_USAGE_EXAMPLE_PATH / "rel_openai" / config_name,
        verbose=False,
    )
