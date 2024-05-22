from ....registry import registry
from typing import Optional, Dict, Any
from .model import AzureMistral
from confection import SimpleFrozenDict


@registry.llm_models("AzureMistral.v1")
def azure_mistral(
    name,
    endpoint,
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=0.0),
    strict: bool = AzureMistral.DEFAULT_STRICT,
    max_tries: int = AzureMistral.DEFAULT_MAX_TRIES,
    interval: float = AzureMistral.DEFAULT_INTERVAL,
    max_request_time: float = AzureMistral.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
):

    return AzureMistral(
        name=name,
        endpoint=endpoint,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )
