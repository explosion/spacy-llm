from typing import Any, Callable, Dict, Iterable, Optional

from confection import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Cohere, Endpoints


@registry.llm_models("spacy.Command.v2")
def cohere_command_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "command",
    strict: bool = Cohere.DEFAULT_STRICT,
    max_tries: int = Cohere.DEFAULT_MAX_TRIES,
    interval: float = Cohere.DEFAULT_INTERVAL,
    max_request_time: float = Cohere.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Cohere instance for 'command' model using REST to prompt API.
    name (str): Name of model to use, e. g. "command" or "command-light".
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    context_length (Optional[int]): Context length for this model. Only necessary for sharding and if no context length
        natively provided by spacy-llm.
    RETURNS (Cohere): Cohere instance for 'command' model.
    """
    return Cohere(
        name=name,
        endpoint=Endpoints.COMPLETION.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Command.v1")
def cohere_command(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "command", "command-light", "command-light-nightly", "command-nightly"
    ] = "command",  # noqa: F821
    strict: bool = Cohere.DEFAULT_STRICT,
    max_tries: int = Cohere.DEFAULT_MAX_TRIES,
    interval: float = Cohere.DEFAULT_INTERVAL,
    max_request_time: float = Cohere.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Cohere instance for 'command' model using REST to prompt API.
    name (Literal["command", "command-light", "command-light-nightly", "command-nightly"]): Model  to use.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Cohere): Cohere instance for 'command' model.
    """
    return Cohere(
        name=name,
        endpoint=Endpoints.COMPLETION.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )
