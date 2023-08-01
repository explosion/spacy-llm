from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Llama2(HuggingFace):
    MODEL_NAMES = Literal[
        "Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf"
    ]  # noqa: F722

    def __init__(
        self,
        name: MODEL_NAMES,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        super().__init__(name=name, config_init=config_init, config_run=config_run)
        # Instantiate GenerationConfig object from config dict.
        self._hf_config_run = transformers.GenerationConfig.from_pretrained(
            self._name,
            **self._config_run,
        )
        # To avoid deprecation warning regarding usage of `max_length`.
        self._hf_config_run.max_new_tokens = self._hf_config_run.max_length

    def init_model(self) -> Any:
        return transformers.pipeline(
            "text-generation",
            model=self._name,
            use_auth_token=True,
            **self._config_init,
        )

    @property
    def hf_account(self) -> str:
        return "meta-llama"

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        return [
            self._model(pr, generation_config=self._hf_config_run)[0]["generated_text"]
            for pr in prompts
        ]

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "trust_remote_code": True,
            },
            default_cfg_run,
        )


@registry.llm_models("spacy.Llama2.v1")
def llama2_hf(
    name: Llama2.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates Llama 2 instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Llama 2 model. Has to be one of Llama2.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Llama2 instance that can execute a set of prompts and return
        the raw responses.
    """
    return Llama2(name=name, config_init=config_init, config_run=config_run)
