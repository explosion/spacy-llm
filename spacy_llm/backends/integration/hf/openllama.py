from typing import Iterable, Callable, Any, Dict, Optional

from spacy.util import SimpleFrozenList, SimpleFrozenDict

from .base import HuggingFaceBackend
from ....compat import transformers, torch
from ....registry.util import registry


class OpenLLaMaBackend(HuggingFaceBackend):
    def __init__(
        self,
        model: str,
        config: Dict[Any, Any],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
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

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_input_ids = [
            self._tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts
        ]

        assert hasattr(self._model, "generate")
        return [
            self._tokenizer.decode(
                self._model.generate(input_ids=tii, **self._config["run"])[0],
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
    def compile_default_config() -> Dict[Any, Any]:
        default_cfg = HuggingFaceBackend.compile_default_config()
        return {
            "init": {**default_cfg["init"], "torch_dtype": torch.float16},
            "run": {"max_new_tokens": 32},
        }


@registry.llm_backends("spacy.OpenLLaMaHF.v1")
def backend_openllama_hf(
    model: str,
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable that can execute a set of prompts and return the raw responses.
    model (str): Name of the HF model.
    config (Dict[Any, Any]): config arguments passed on to the initialization of transformers.pipeline instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    return OpenLLaMaBackend(
        model=model,
        config=config,
    )
