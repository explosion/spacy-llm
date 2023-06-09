from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict, SimpleFrozenList

from ....compat import torch, transformers
from ....registry.util import registry
from .base import HuggingFaceBackend


class OpenLLaMaBackend(HuggingFaceBackend):
    def __init__(
        self,
        model: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
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
    def supported_models(self) -> Iterable[str]:
        return SimpleFrozenList(
            [
                "openlm-research/open_llama_3b_350bt_preview",
                "openlm-research/open_llama_3b_600bt_preview",
                "openlm-research/open_llama_7b_400bt_preview",
                "openlm-research/open_llama_7b_700bt_preview",
            ]
        )

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFaceBackend.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "torch_dtype": torch.float16,
            },
            {**default_cfg_run, "max_new_tokens": 32},
        )


@registry.llm_backends("spacy.OpenLLaMa_HF.v1")
def backend_openllama_hf(
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
    return OpenLLaMaBackend(
        model=model,
        config_init=config_init,
        config_run=config_run,
    )
