import warnings
from typing import Any, Collection, Dict

from thinc.compat import has_torch_cuda_gpu

from ...compat import has_accelerate, has_torch, has_transformers, torch


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


def _check_model(model: str, supported_models: Collection[str]) -> None:
    if model not in supported_models:
        raise ValueError(
            f"Model '{model}' is not supported - select one of {supported_models} instead"
        )


def _compile_default_config() -> Dict[Any, Any]:
    default_cfg: Dict[str, Any] = {
        # Loads a custom pipeline from https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py
        # cf also https://huggingface.co/databricks/dolly-v2-12b
        "trust_remote_code": True,
    }

    if has_torch:
        default_cfg["torch_dtype"] = torch.bfloat16
        if has_torch_cuda_gpu:
            # this ensures it fails explicitely when GPU is not enabled or sufficient
            default_cfg["device"] = "cuda:0"
        elif has_accelerate:
            # accelerate will distribute the layers depending on availability on GPU/CPU/hard drive
            default_cfg["device_map"] = "auto"
            warnings.warn(
                "Couldn't find a CUDA GPU, so the setting 'device_map:auto' will be used, which may result "
                "in the LLM being loaded (partly) on the CPU or even the hard disk, which may be slow. "
                "Install cuda to be able to load and run the LLM on the GPU instead."
            )
        else:
            raise ValueError(
                "Install CUDA to load and run the LLM on the GPU, or install 'accelerate' to dynamically "
                "distribute the LLM on the CPU or even the hard disk. The latter may be slow."
            )
    return default_cfg
