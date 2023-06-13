from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict, SimpleFrozenList

from ....compat import transformers
from ....registry.util import registry
from . import HuggingFaceBackend


class DollyBackend(HuggingFaceBackend):
    def init_model(self) -> Any:
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        return transformers.pipeline(model=self._model_name, **self._config_init)

    @property
    def supported_models(self) -> Iterable[str]:
        return SimpleFrozenList(
            [
                "databricks/dolly-v2-3b",
                "databricks/dolly-v2-7b",
                "databricks/dolly-v2-12b",
            ]
        )

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        """Queries Dolly HF model.
        pipeline (transformers.pipeline): Transformers pipeline to query.
        prompts (Iterable[str]): Prompts to query Dolly model with.
        RETURNS (Iterable[str]): Prompt responses.
        """
        return [
            self._model(pr, **self._config_run)[0]["generated_text"] for pr in prompts
        ]

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFaceBackend.compile_default_configs()
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


@registry.llm_backends("spacy.DollyHF.v1")
@registry.llm_backends("spacy.Dolly_HF.v1")
def backend_dolly_hf(
    model: str,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable that can execute a set of prompts and return the raw responses.
    model (str): Name of the HF model.
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    return DollyBackend(
        model=model,
        config_init=config_init,
        config_run=config_run,
    )
