import copy

import pytest
import spacy
from confection import Config  # type: ignore[import]
from thinc.compat import has_torch_cuda_gpu

from ...compat import torch

_PIPE_CFG = {
    "model": {
        "@llm_models": "spacy.Mistral.v1",
        "name": "Mistral-7B-v0.1",
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}

_NLP_CONFIG = """

[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.NoOp.v1"

[components.llm.model]
@llm_models = "spacy.Mistral.v1"
name = "Mistral-7B-v0.1"
"""


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init():
    """Test initialization and simple run."""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(_PIPE_CFG)
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init_from_config():
    orig_config = Config().from_str(_NLP_CONFIG)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_invalid_model():
    orig_config = Config().from_str(_NLP_CONFIG)
    config = copy.deepcopy(orig_config)
    config["components"]["llm"]["model"]["name"] = "x"
    with pytest.raises(ValueError, match="unexpected value; permitted"):
        spacy.util.load_model_from_config(config, auto_fill=True)
    torch.cuda.empty_cache()
