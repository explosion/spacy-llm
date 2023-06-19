from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict

from ...compat import Literal, torch, transformers
from ...registry.util import registry
from .base import HuggingFace


class OpenLLaMa(HuggingFace):
    def __init__(
        self,
        variant: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._device: Optional[str] = None
        super().__init__(
            variant=variant, config_init=config_init, config_run=config_run
        )

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        # Initialize tokenizer and model.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        init_cfg = self._config_init
        if "device" in init_cfg:
            self._device = init_cfg.pop("device")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, **init_cfg
        )

        if self._device:
            model.to(self._device)

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_input_ids = [
            self._tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts
        ]
        if self._device:
            tokenized_input_ids = [tii.to(self._device) for tii in tokenized_input_ids]

        assert hasattr(self._model, "generate")
        return [
            self._tokenizer.decode(
                self._model.generate(input_ids=tii, **self._config_run)[0],
            )
            for tii in tokenized_input_ids
        ]

    @property
    def model_name(self) -> str:
        return f"openlm-research/open_llama_{self._variant}_preview"

    @staticmethod
    def get_model_variants() -> Iterable[str]:
        return ["3b_350bt", "3b_600bt", "7b_400bt", "7b_600bt"]

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "torch_dtype": torch.float16,
            },
            {**default_cfg_run, "max_new_tokens": 32},
        )


@registry.llm_models("spacy.OpenLLaMa.HF.v1")
def openllama_hf(
    variant: Literal["3b_350bt", "3b_600bt", "7b_400bt", "7b_600bt"],  # noqa: F722
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates OpenLLaMa instance that can execute a set of prompts and return the raw responses.
    variant (Literal): Name of the StableLM model variant. Has to be one of OpenLLaMa.get_model_variants().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): OpenLLaMa instance that can execute a set of prompts and return
        the raw responses.
    """
    return OpenLLaMa(variant=variant, config_init=config_init, config_run=config_run)
