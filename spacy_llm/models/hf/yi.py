from typing import Any, Dict, Iterable, List, Optional, Tuple

from confection import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Yi(HuggingFace):
    MODEL_NAMES = Literal[  # noqa: F722
        "Yi-34B",
        "Yi-34B-Chat-8bits",
        "Yi-6B-Chat",
        "Yi-6B",
        "Yi-6B-200K",
        "Yi-34B-Chat",
        "Yi-34B-Chat-4bits",
        "Yi-34B-200K",
    ]

    def __init__(
        self,
        name: MODEL_NAMES,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
        context_length: int,
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
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._name, use_fast=False
        )
        init_cfg = self._config_init
        device: Optional[str] = None
        if "device" in init_cfg:
            device = init_cfg.pop("device")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, **init_cfg, resume_download=True
        ).eval()
        if device:
            model.to(device)

        return model

    @property
    def hf_account(self) -> str:
        return "01-ai"

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:  # type: ignore[override]
        assert hasattr(self._model, "generate")
        assert hasattr(self._tokenizer, "apply_chat_template")
        assert self._tokenizer

        responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            prompts_for_doc = list(prompts_for_doc)

            tokenized_input_ids = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                for prompt in prompts_for_doc
            ]
            tokenized_input_ids = [
                tp.to(self._model.device) for tp in tokenized_input_ids
            ]

            # response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
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

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return {**default_cfg_init, **{"torch_dtype": "auto"}}, default_cfg_run


@registry.llm_models("spacy.Yi.v1")
def yi_hf(
    name: Yi.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Yi:
    """Generates Yi instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Mistral model. Has to be one of Mistral.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Yi): Yi instance that can execute a set of prompts and return the raw responses.
    """
    return Yi(
        name=name,
        config_init=config_init,
        config_run=config_run,
        context_length=200000 if "200K" in name else 32000,
    )
