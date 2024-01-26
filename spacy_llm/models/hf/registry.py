from typing import Any, Callable, Dict, Iterable, Optional

from confection import SimpleFrozenDict

from ...registry import registry
from .dolly import Dolly
from .falcon import Falcon
from .llama2 import Llama2
from .mistral import Mistral
from .openllama import OpenLLaMA
from .stablelm import StableLM


@registry.llm_models("spacy.HuggingFace.v1")
def huggingface_v1(
    name: str,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns HuggingFace model instance.
    name (str): Name of model to use.
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Model instance that can execute a set of prompts and return
        the raw responses.
    """
    model_context_lengths = {
        Dolly: 2048,
        Falcon: 2048,
        Llama2: 4096,
        Mistral: 8000,
        OpenLLaMA: 2048,
        StableLM: 4096,
    }

    for model_cls, context_length in model_context_lengths.items():
        model_names = getattr(model_cls, "MODEL_NAMES")
        if model_names and name in model_names.__args__:
            return model_cls(
                name=name,
                config_init=config_init,
                config_run=config_run,
                context_length=context_length,
            )

    raise ValueError(
        f"Name {name} could not be associated with any of the supported models. Please check "
        f"https://spacy.io/api/large-language-models#models-hf to ensure the specified model name is correct."
    )
