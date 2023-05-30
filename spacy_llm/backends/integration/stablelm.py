from typing import Iterable, Callable, Any, Dict, Optional, Tuple

from spacy.util import SimpleFrozenList, SimpleFrozenDict

from .base import HuggingFaceBackend, _PromptType, _ResponseType
from ...compat import transformers, torch
from ...registry.util import registry


class StableLMHFBackend(HuggingFaceBackend):
    _SYSTEM_PROMPT = """
<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    class StopOnTokens(transformers.StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def __init__(
        self,
        query: Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]],
        model: str,
        config: Dict[Any, Any],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._is_tuned = "tuned" in model
        super().__init__(query=query, model=model, config=config)

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model, **self._config
        )
        model.half().cuda()

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        return self.query(  # type: ignore[return-value]
            self._integration,
            [
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
            ],
        )

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


def query_stablelm(
    model: Tuple["transformers.AutoModelForCausalLM", "transformers.AutoTokenizer"],
    prompts: Iterable[Dict[Any, Any]],
) -> Iterable[str]:
    """Queries StableLM HF model.
    model (Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]): HF model and tokenizer (to use on
        model in-/output).
    prompts (Iterable[str]): Prompts to query Dolly model with.
    RETURNS (Iterable[str]): Prompt responses.
    """
    (_model, _tokenizer) = model
    assert hasattr(_model, "generate")
    return [
        _tokenizer.decode(
            _model.generate(
                **prompt,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
            )[0],
            skip_special_tokens=True,
        )
        for prompt in prompts
    ]


@registry.llm_backends("spacy.OpenLLaMaHF.v1")
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
        query=query_stablelm,  # type: ignore[arg-type]
        model=model,
        config=config,
    )
