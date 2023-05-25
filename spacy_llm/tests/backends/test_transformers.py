import copy

import pytest
import spacy
from confection import Config
from thinc.compat import has_torch_cuda_gpu

from spacy_llm.backends.hf import dolly_supported_models, supported_models

backend_and_model = [("spacy.DollyHF.v1", model) for model in dolly_supported_models]
backend_and_model.extend([("spacy.HF.v1", model) for model in supported_models])


@pytest.fixture(params=backend_and_model)
def config(request: pytest.FixtureRequest) -> Config:

    backend, model = request.param

    return {
        "backend": {
            "@llm_backends": backend,
            "model": model,
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
@llm_backends = "spacy.HF.v1"
model = "databricks/dolly-v2-3b"
"""


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_init(config):
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=config)
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
