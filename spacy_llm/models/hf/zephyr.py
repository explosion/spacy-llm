from typing import Any, Dict, Iterable, List, Optional, Tuple

from confection import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Zephyr(HuggingFace):
    MODEL_NAMES = Literal["zephyr-7b-beta"]  # noqa: F722

    def __init__(
        self,
        name: MODEL_NAMES,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
        context_length: int,
    ):
        super().__init__(
            name=name,
            config_init=config_init,
            config_run=config_run,
            context_length=context_length,
        )

        # Instantiate GenerationConfig object from config dict.
        self._hf_config_run = transformers.GenerationConfig.from_pretrained(
            self._name, **self._config_run
        )
        # To avoid deprecation warning regarding usage of `max_length`.
        self._hf_config_run.max_new_tokens = self._hf_config_run.max_length

    def init_model(self) -> Any:
        return transformers.pipeline(
            "text-generation",
            model=self._name,
            return_full_text=False,
            **self._config_init
        )

    @property
    def hf_account(self) -> str:
        return "HuggingFaceH4"

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:  # type: ignore[override]
        responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            formatted_prompts_for_doc = [
                self._model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for prompt in prompts_for_doc
            ]

            responses.append(
                [
                    self._model(prompt, generation_config=self._hf_config_run)[0][
                        "generated_text"
                    ]
                    .replace("<|assistant|>", "")
                    .strip("\n")
                    for prompt in formatted_prompts_for_doc
                ]
            )

        return responses

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return default_cfg_init, {
            **default_cfg_run,
            **{
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
            },
        }


@registry.llm_models("spacy.Zephyr.v1")
def zephyr_hf(
    name: Zephyr.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Zephyr:
    """Generates Zephyr instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Zephyr model. Has to be one of Zephyr.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Zephyr): Zephyr instance that can execute a set of prompts and return the raw responses.
    """
    return Zephyr(
        name=name, config_init=config_init, config_run=config_run, context_length=8000
    )
