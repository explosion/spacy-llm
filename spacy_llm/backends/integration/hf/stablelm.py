from typing import Iterable, Callable, Any, Dict, Optional

from spacy.util import SimpleFrozenList, SimpleFrozenDict

from .base import HuggingFaceBackend
from ....compat import transformers, torch, has_transformers
from ....registry.util import registry

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


class StableLMHFBackend(HuggingFaceBackend):
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
        config: Dict[Any, Any],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._is_tuned = "tuned" in model
        super().__init__(model=model, config=config)

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        # Initialize tokenizer and model.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_name, **self._config["init"]
        )
        model.half().cuda()

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_prompts = [
            self._tokenizer(prompt, return_tensors="pt").to("cuda")
            for prompt in (
                # Add prompt formatting for tuned model.
                prompts
                if not self._is_tuned
                else [
                    f"{StableLMHFBackend._SYSTEM_PROMPT}<|USER|>{prompt}<|ASSISTANT|>"
                    for prompt in prompts
                ]
            )
        ]

        assert hasattr(self._model, "generate")
        return [
            self._tokenizer.decode(
                self._model.generate(**prompt, **self._config["run"])[0],
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
    def compile_default_config() -> Dict[Any, Any]:
        default_cfg = HuggingFaceBackend.compile_default_config()
        return {
            "init": {k: v for k, v in default_cfg["init"].items() if k != "device"},
            "run": {
                "max_new_tokens": 64,
                "temperature": 0.7,
                "do_sample": True,
            },
        }


@registry.llm_backends("spacy.StableLMHF.v1")
def backend_stablelm_hf(
    model: str,
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable that can execute a set of prompts and return the raw responses.
    model (str): Name of the HF model.
    config (Dict[Any, Any]): config arguments passed on to the initialization of transformers.pipeline instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    return StableLMHFBackend(
        model=model,
        config=config,
    )
