import copy

import pytest
import spacy
from confection import Config  # type: ignore[import]
from thinc.compat import has_torch_cuda_gpu

_PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.OpenLLaMa_HF.v1",
        "model": "openlm-research/open_llama_3b_350bt_preview",
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

[components.llm.backend]
@llm_backends = spacy.OpenLLaMa_HF.v1
model = "openlm-research/open_llama_3b_350bt_preview"
"""


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init():
    """Test initialization and simple run."""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=_PIPE_CFG)
    nlp("This is a test.")


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init_with_set_config():
    """Test initialization and simple run with changed config."""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(_PIPE_CFG)
    cfg["backend"]["config_run"] = {"max_new_tokens": 32}
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init_from_config():
    orig_config = Config().from_str(_NLP_CONFIG)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_invalid_model():
    orig_config = Config().from_str(_NLP_CONFIG)
    config = copy.deepcopy(orig_config)
    config["components"]["llm"]["backend"]["model"] = "anything-else"
    with pytest.raises(ValueError, match="is not supported"):
        spacy.util.load_model_from_config(config, auto_fill=True)
