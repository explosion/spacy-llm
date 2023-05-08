import warnings
from typing import Any, Callable, Dict, Iterable

import spacy
from thinc.compat import has_torch_cuda_gpu

from ..compat import (has_accelerate, has_torch, has_transformers, torch,
                      transformers)

DEFAULT_HF_DICT = {
    "trust_remote_code": True,
}

if has_torch:
    DEFAULT_HF_DICT["torch_dtype"] = torch.bfloat16
    if has_torch_cuda_gpu:
        # this ensures it fails explicitely when GPU is not enabled or sufficient
        DEFAULT_HF_DICT["device"] = "cuda:0"
    elif has_accelerate:
        # accelerate will distribute the layers depending on availability on GPU/CPU/hard drive
        DEFAULT_HF_DICT["device_map"] = "auto"
        warnings.warn(
            "Couldn't find a CUDA GPU, so the setting 'device_map:auto' will be used, which may result "
            "in the LLM being loaded (partly) on the CPU or even the hard disk, which may be slow. "
            "Install cuda to be able to load and run the LLM on the GPU instead."
        )
    else:
        warnings.warn(
            "Install CUDA to load and run the LLM on the GPU, or install 'accelerate' to dynamically "
            "distribute the LLM on the CPU or even the hard disk. The latter may be slow."
        )


def _check_installation() -> None:
    """Checks whether the required external libraries are installed. Raises an error otherwise."""
    if not has_torch:
        raise ValueError(
            "The HF backend requires `torch` to be installed, which it is not. See "
            "https://pytorch.org/ for installation instructions."
        )
    if not has_transformers:
        raise ValueError(
            "The HF backend requires `transformers` to be installed, which it is not. See "
            "https://huggingface.co/docs/transformers/installation for installation instructions."
        )


@spacy.registry.llm_backends("spacy.HF.v1")
def backend_hf(
    model: str,
    config: Dict[Any, Any] = DEFAULT_HF_DICT,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Pipeline instance.
    model (str): Name of the HF model
    config (Dict[Any, Any]): config arguments passed on to the initialization of transformers.pipeline instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    _check_installation()
    llm_pipeline = transformers.pipeline(model=model, **config)

    def query(prompts: Iterable[str]) -> Iterable[str]:
        return [llm_pipeline(pr)[0]["generated_text"] for pr in prompts]

    return query
