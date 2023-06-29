from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from spacy.util import SimpleFrozenDict

from ...compat import Literal, transformers
from ...registry.util import registry
from .base import HuggingFace


class Falcon(HuggingFace):
    MODEL_NAMES = Literal[
        "falcon-rw-1b", "falcon-7b", "falcon-7b-instruct", "falcon-40b-instruct"
    ]  # noqa: F722

    def __init__(
        self,
        name: MODEL_NAMES,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._device: Optional[str] = None
        super().__init__(name=name, config_init=config_init, config_run=config_run)
        assert isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase)
        self._config_run["eos_token_id"] = self._tokenizer.eos_token_id

    def init_model(self) -> Any:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._name)
        return transformers.pipeline(
            "text-generation",
            model=self._name,
            tokenizer=self._tokenizer,
            **self._config_init,
        )

    @property
    def hf_account(self) -> str:
        return "tiiuae"

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        return [
            self._model(pr, **self._config_run)[0]["generated_text"] for pr in prompts
        ]

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "trust_remote_code": True,
            },
            {
                **default_cfg_run,
                "max_length": 200,
                "do_sample": True,
                "top_k": 10,
                "num_return_sequences": 1,
            },
        )


@registry.llm_models("spacy.Falcon.v1")
def falcon_hf(
    name: Falcon.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates Falcon instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Falcon model. Has to be one of Falcon.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Falcon instance that can execute a set of prompts and return
        the raw responses.
    """
    return Falcon(name=name, config_init=config_init, config_run=config_run)
