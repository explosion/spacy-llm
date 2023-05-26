from typing import Any, Callable, Dict, Iterable

from spacy.util import SimpleFrozenDict, SimpleFrozenList

from ....compat import transformers
from ....registry.util import registry
from .. import Backend
from .util import _check_installation, _check_model, _compile_default_config

supported_models = SimpleFrozenList(
    ["databricks/dolly-v2-3b", "databricks/dolly-v2-7b", "databricks/dolly-v2-12b"]
)


def query_dolly(
    pipeline: "transformers.pipeline", prompts: Iterable[str]
) -> Iterable[str]:
    """Queries Dolly HF model.
    pipeline (transformers.pipeline): Transformers pipeline to query.
    prompts (Iterable[str]): Prompts to query Dolly model with.
    RETURNS (Iterable[str]): Prompt responses.
    """
    return [pipeline(pr)[0]["generated_text"] for pr in prompts]


@registry.llm_backends("spacy.DollyHF.v1")
def backend_dolly_hf(
    model: str,
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable that can execute a set of prompts and return the raw responses.
    model (str): Name of the HF model.
    config (Dict[Any, Any]): config arguments passed on to the initialization of transformers.pipeline instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    _check_installation()
    _check_model(model, supported_models=supported_models)

    if not config or len(config) == 0:
        config = _compile_default_config()

    return Backend(
        integration=transformers.pipeline(model=model, **config), query=query_dolly  # type: ignore[arg-type]
    )
