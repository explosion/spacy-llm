from typing import Tuple

import pytest
import spacy
from thinc.compat import has_torch_cuda_gpu

from spacy_llm.compat import has_accelerate

_PIPE_CFG = {
    "model": {
        "@llm_models": "",
        "name": "",
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
    "save_io": True,
}


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
@pytest.mark.parametrize(
    "model", (("spacy.Dolly.v1", "dolly-v2-3b"), ("spacy.Llama2.v1", "Llama-2-7b-hf"))
)
def test_device_config_conflict(model: Tuple[str, str]):
    """Test initialization and simple run."""
    nlp = spacy.blank("en")
    model, name = model
    cfg = {**_PIPE_CFG, **{"model": {"@llm_models": model, "name": name}}}

    # Set device only.
    cfg["model"]["config_init"] = {"device": "cpu"}  # type: ignore[index]
    nlp.add_pipe("llm", name="llm1", config=cfg)

    # Set device_map only.
    cfg["model"]["config_init"] = {"device_map": "auto"}  # type: ignore[index]
    if has_accelerate:
        nlp.add_pipe("llm", name="llm2", config=cfg)
    else:
        with pytest.raises(ImportError, match="requires Accelerate"):
            nlp.add_pipe("llm", name="llm2", config=cfg)

    # Set device_map and device.
    cfg["model"]["config_init"] = {"device_map": "auto", "device": "cpu"}  # type: ignore[index]
    with pytest.warns(UserWarning, match="conflicting arguments"):
        if has_accelerate:
            nlp.add_pipe("llm", name="llm3", config=cfg)
        else:
            with pytest.raises(ImportError, match="requires Accelerate"):
                nlp.add_pipe("llm", name="llm3", config=cfg)
