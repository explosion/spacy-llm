from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from confection import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Phi2(HuggingFace):
    MODEL_NAMES = Literal["phi-2"]  # noqa: F722

    def __init__(
        self,
        name: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
        context_length: Optional[int],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        super().__init__(
            name=name,
            config_init=config_init,
            config_run=config_run,
            context_length=context_length,
        )

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        # Initialize tokenizer and model.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._name, trust_remote_code=True
        )
        init_cfg = self._config_init
        device: Optional[str] = None
        if "device" in init_cfg:
            device = init_cfg.pop("device")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, **init_cfg
        )
        if device:
            model.to(device)

        return model

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:  # type: ignore[override]
        assert callable(self._tokenizer)
        responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            tokenized_input_ids = [
                self._tokenizer(
                    prompt, return_tensors="pt", return_attention_mask=False
                ).input_ids
                for prompt in prompts_for_doc
            ]
            tokenized_input_ids = [
                tii.to(self._model.device) for tii in tokenized_input_ids
            ]

            assert hasattr(self._model, "generate")
            responses.append(
                [
                    self._tokenizer.decode(
                        self._model.generate(input_ids=tii, **self._config_run)[
                            :, tii.shape[1] :
                        ][0],
                    )
                    for tii in tokenized_input_ids
                ]
            )

        return responses

    @property
    def hf_account(self) -> str:
        return "microsoft"

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # See https://huggingface.co/microsoft/phi-2#sample-code for recommended setting combinations.
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "torch_dtype": "auto",
                "device_map": "cuda",
                "trust_remote_code": True,
            },
            {
                **default_cfg_run,
                "max_new_tokens": 200,
            },
        )


@registry.llm_models("spacy.Phi-2.v1")
def phi2_hf(
    name: Phi2.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Generates OpenLLaMA instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the OpenLLaMA model. Has to be one of OpenLLaMA.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): OpenLLaMA instance that can execute a set of prompts and return
        the raw responses.
    """
    return Phi2(
        name=name, config_init=config_init, config_run=config_run, context_length=2048
    )
