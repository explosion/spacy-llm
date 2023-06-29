from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from . import HuggingFace


class Dolly(HuggingFace):
    MODEL_NAMES = Literal["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]  # noqa: F722

    def init_model(self) -> Any:
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        return transformers.pipeline(model=self._name, **self._config_init)

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        """Queries Dolly HF model.
        pipeline (transformers.pipeline): Transformers pipeline to query.
        prompts (Iterable[str]): Prompts to query Dolly model with.
        RETURNS (Iterable[str]): Prompt responses.
        """
        return [
            self._model(pr, **self._config_run)[0]["generated_text"] for pr in prompts
        ]

    @property
    def hf_account(self) -> str:
        return "databricks"

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                # Loads a custom pipeline from
                # https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py
                # cf also https://huggingface.co/databricks/dolly-v2-12b
                "trust_remote_code": True,
            },
            default_cfg_run,
        )


@registry.llm_models("spacy.Dolly.v1")
def dolly_hf(
    name: Dolly.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates Dolly instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Dolly model. Has to be one of Dolly.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Dolly instance that can execute a set of prompts and return
        the raw responses.
    """
    return Dolly(name=name, config_init=config_init, config_run=config_run)
