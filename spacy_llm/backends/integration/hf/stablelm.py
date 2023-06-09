from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict, SimpleFrozenList

from ....compat import has_transformers, torch, transformers
from ....registry.util import registry
from .base import HuggingFaceBackend

if has_transformers:

    class _StopOnTokens(transformers.StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False


class StableLMBackend(HuggingFaceBackend):
    _SYSTEM_PROMPT = """
<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    def __init__(
        self,
        model: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._is_tuned = "tuned" in model
        self._device: Optional[str] = None
        super().__init__(model=model, config_init=config_init, config_run=config_run)

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        # Initialize tokenizer and model.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
        init_cfg = self._config_init
        if "device" in init_cfg:
            self._device = init_cfg.pop("device")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_name, **init_cfg
        )

        if self._device:
            model.half().to(self._device)

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_prompts = [
            self._tokenizer(prompt, return_tensors="pt")
            for prompt in (
                # Add prompt formatting for tuned model.
                prompts
                if not self._is_tuned
                else [
                    f"{StableLMBackend._SYSTEM_PROMPT}<|USER|>{prompt}<|ASSISTANT|>"
                    for prompt in prompts
                ]
            )
        ]
        if self._device:
            tokenized_prompts = [tp.to(self._device) for tp in tokenized_prompts]

        assert hasattr(self._model, "generate")
        return [
            self._tokenizer.decode(
                self._model.generate(**prompt, **self._config_run)[0],
                skip_special_tokens=True,
            )
            for prompt in tokenized_prompts
        ]

    @property
    def supported_models(self) -> Iterable[str]:
        return SimpleFrozenList(
            [
                "stabilityai/stablelm-base-alpha-3b",
                "stabilityai/stablelm-base-alpha-7b",
                "stabilityai/stablelm-tuned-alpha-3b",
                "stabilityai/stablelm-tuned-alpha-7b",
            ]
        )

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFaceBackend.compile_default_configs()
        return (
            default_cfg_init,
            {
                **default_cfg_run,
                "max_new_tokens": 64,
                "temperature": 0.7,
                "do_sample": True,
            },
        )


@registry.llm_backends("spacy.StableLM_HF.v1")
def backend_stablelm_hf(
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
    return StableLMBackend(
        model=model,
        config_init=config_init,
        config_run=config_run,
    )
