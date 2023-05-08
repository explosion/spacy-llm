from typing import Any, Callable, Dict, Iterable

import spacy

from ..compat import has_torch, has_transformers, torch, transformers

DEFAULT_HF_DICT = {
    "trust_remote_code": True,
    "device": "cuda:0",
}

if has_torch:
    DEFAULT_HF_DICT["torch_dtype"] = torch.bfloat16


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
