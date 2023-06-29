from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict

from ...compat import Literal, has_transformers, torch, transformers
from ...registry.util import registry
from .base import HuggingFace

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


class StableLM(HuggingFace):
    MODEL_NAMES = Literal[
        "stablelm-base-alpha-3b",  # noqa: F722
        "stablelm-base-alpha-7b",  # noqa: F722
        "stablelm-tuned-alpha-3b",  # noqa: F722
        "stablelm-tuned-alpha-7b",  # noqa: F722
    ]
    _SYSTEM_PROMPT = """
<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    def __init__(
        self,
        name: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._is_tuned = "tuned" in name
        self._device: Optional[str] = None
        super().__init__(name=name, config_init=config_init, config_run=config_run)

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._name)
        init_cfg = self._config_init
        if "device" in init_cfg:
            self._device = init_cfg.pop("device")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, **init_cfg
        )

        if self._device:
            model.half().to(self._device)

        return model

    @property
    def hf_account(self) -> str:
        return "stabilityai"

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_prompts = [
            self._tokenizer(prompt, return_tensors="pt")
            for prompt in (
                # Add prompt formatting for tuned model.
                prompts
                if not self._is_tuned
                else [
                    f"{StableLM._SYSTEM_PROMPT}<|USER|>{prompt}<|ASSISTANT|>"
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

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            default_cfg_init,
            {
                **default_cfg_run,
                "max_new_tokens": 64,
                "temperature": 0.7,
                "do_sample": True,
            },
        )


@registry.llm_models("spacy.StableLM.v1")
def stablelm_hf(
    name: StableLM.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates StableLM instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the StableLM model. Has to be one of StableLM.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): StableLM instance that can execute a set of prompts and return
        the raw responses.
    """
    if name not in StableLM.get_model_names():
        raise ValueError(
            f"Expected one of {StableLM.get_model_names()}, but received {name}."
        )
    return StableLM(
        name=name,
        config_init=config_init,
        config_run=config_run,
    )
