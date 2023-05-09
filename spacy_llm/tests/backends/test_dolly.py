import copy

import spacy
import pytest
from confection import Config  # type: ignore[import]
from thinc.compat import has_torch_cuda_gpu

_PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.DollyHF.v1",
        "model": "databricks/dolly-v2-3b",
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
@llm_backends = "spacy.DollyHF.v1"
model = "databricks/dolly-v2-3b"
"""


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=_PIPE_CFG)
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
    config["components"]["llm"]["backend"]["model"] = "dolly-the-sheep"
    with pytest.raises(ValueError, match="is not supported"):
        spacy.util.load_model_from_config(config, auto_fill=True)
