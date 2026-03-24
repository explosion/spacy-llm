from typing import Any, Dict, Optional

from confection import SimpleFrozenDict

from ....registry import registry
from .model import Endpoints, MiniMax

_DEFAULT_TEMPERATURE = 0.0


@registry.llm_models("spacy.MiniMax.v1")
def minimax_v1(
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    name: str = "MiniMax-M2.5",
    strict: bool = MiniMax.DEFAULT_STRICT,
    max_tries: int = MiniMax.DEFAULT_MAX_TRIES,
    interval: float = MiniMax.DEFAULT_INTERVAL,
    max_request_time: float = MiniMax.DEFAULT_MAX_REQUEST_TIME,
    endpoint: Optional[str] = None,
    context_length: Optional[int] = None,
) -> MiniMax:
    """Returns MiniMax instance for MiniMax models using REST to prompt API.

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    name (str): Model name to use. Can be any model name supported by the MiniMax API - e.g.
        'MiniMax-M2.5', 'MiniMax-M2.5-highspeed', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed'.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response.
        If False, the API error responses are returned by __call__(), but no error will be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2
        exponential backoff at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising
        an exception.
    endpoint (Optional[str]): Endpoint to set. Defaults to standard endpoint.
    context_length (Optional[int]): Context length for this model. Only necessary for sharding and
        if no context length natively provided by spacy-llm.
    RETURNS (MiniMax): MiniMax instance.

    DOCS: https://platform.minimaxi.com/
    """
    return MiniMax(
        name=name,
        endpoint=endpoint or Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )
