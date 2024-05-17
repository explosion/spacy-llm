from typing import Any, Callable, Dict, Iterable, List, Optional

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
        context_length: Optional[int],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._is_instruct = "instruct" in name
        super().__init__(
            name=name,
            config_init=config_init,
            config_run=config_run,
            context_length=context_length,
        )

        assert isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase)

        # Instantiate GenerationConfig object from config dict.
        self._hf_config_run = transformers.GenerationConfig.from_pretrained(
            self._name, **self._config_run
        )
        # To avoid deprecation warning regarding usage of `max_length`.
        self._hf_config_run.max_new_tokens = self._hf_config_run.max_length

    def init_model(self) -> Any:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._name)
        init_cfg = self._config_init
        device: Optional[str] = None
        if "device" in init_cfg:
            device = init_cfg.pop("device")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, **init_cfg, resume_download=True
        )
        if device:
            model.to(device)

        return model

    @property
    def hf_account(self) -> str:
        return "mistralai"

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:  # type: ignore[override]
        assert callable(self._tokenizer)
        assert hasattr(self._model, "generate")
        assert hasattr(self._tokenizer, "batch_decode")
        responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            prompts_for_doc = list(prompts_for_doc)

            tokenized_input_ids = [
                self._tokenizer(
                    prompt if not self._is_instruct else f"<s>[INST] {prompt} [/INST]",
                    return_tensors="pt",
                ).input_ids
                for prompt in prompts_for_doc
            ]
            tokenized_input_ids = [
                tp.to(self._model.device) for tp in tokenized_input_ids
            ]

            responses.append(
                [
                    self._tokenizer.decode(
                        self._model.generate(
                            input_ids=tok_ii, generation_config=self._hf_config_run
                        )[:, tok_ii.shape[1] :][0],
                        skip_special_tokens=True,
                    )
                    for tok_ii in tokenized_input_ids
                ]
            )

        return responses


@registry.llm_models("spacy.Mistral.v1")
def mistral_hf(
    name: Mistral.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Generates Mistral instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Mistral model. Has to be one of Mistral.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Mistral instance that can execute a set of prompts and return
        the raw responses.
    """
    return Mistral(
        name=name, config_init=config_init, config_run=config_run, context_length=8000
    )
