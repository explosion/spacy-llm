from pathlib import Path

import pytest

from ..ner_dolly import run_ner_dolly_pipeline
from ..textcat_openai import run_textcat_openai_pipeline

# from thinc.compat import has_torch_cuda_gpu

_USAGE_EXAMPLE_PATH = Path(__file__).parent.parent


# @pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize(
    "config_name", ("dolly_ner_fewshot.cfg", "dolly_ner_zeroshot.cfg")
)
def test_ner_dolly(config_name: str):
    """Test NER Dolly usage example.
    config_name (str): Name of config file to use.
    """
    run_ner_dolly_pipeline.run_pipeline(
        "text", _USAGE_EXAMPLE_PATH / "ner_dolly" / config_name, False
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
        "text", _USAGE_EXAMPLE_PATH / "textcat_openai" / config_name, False
    )
