from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from confection import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Mistral(HuggingFace):
    MODEL_NAMES = Literal["Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1"]  # noqa: F722

    def __init__(
        self,
        name: MODEL_NAMES,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._device: Optional[str] = None
        self._is_instruct = "instruct" in name
        super().__init__(name=name, config_init=config_init, config_run=config_run)

        assert isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase)
        # self._config_run["pad_token_id"] = self._tokenizer.pad_token_id

        # Instantiate GenerationConfig object from config dict.
        self._hf_config_run = transformers.GenerationConfig.from_pretrained(
            self._name, **self._config_run
        )
        # To avoid deprecation warning regarding usage of `max_length`.
        self._hf_config_run.max_new_tokens = self._hf_config_run.max_length

    def init_model(self) -> Any:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._name)
        init_cfg = self._config_init
        if "device" in init_cfg:
            self._device = init_cfg.pop("device")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, **init_cfg, resume_download=True
        )
        if self._device:
            model.to(self._device)

        return model

    @property
    def hf_account(self) -> str:
        return "mistralai"

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        assert hasattr(self._model, "generate")
        assert hasattr(self._tokenizer, "batch_decode")
        prompts = list(prompts)

        tokenized_prompts = [
            self._tokenizer(
                prompt if not self._is_instruct else f"<s>[INST] {prompt} [/INST]",
                return_tensors="pt",
            )
            for prompt in prompts
        ]
        if self._device:
            tokenized_prompts = [tp.to(self._device) for tp in tokenized_prompts]

        x = [
            self._tokenizer.decode(
                self._model.generate(
                    **tok_prompt, generation_config=self._hf_config_run
                )[0],
                skip_special_tokens=True,
            )
            .replace(prompt, "")
            .strip()
            for tok_prompt, prompt in zip(tokenized_prompts, prompts)
        ]
        return x

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                # "trust_remote_code": True,
            },
            default_cfg_run,
        )


@registry.llm_models("spacy.Mistral.v1")
def mistral_hf(
    name: Mistral.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates Mistral instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Falcon model. Has to be one of Falcon.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Falcon instance that can execute a set of prompts and return
        the raw responses.
    """
    return Mistral(name=name, config_init=config_init, config_run=config_run)
