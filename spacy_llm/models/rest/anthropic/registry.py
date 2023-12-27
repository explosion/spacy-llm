from typing import Any, Callable, Dict, Iterable, Optional

from confection import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Anthropic, Endpoints


@registry.llm_models("spacy.Claude-2.v2")
def anthropic_claude_2_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-2",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Anthropic:
    """Returns Anthropic instance for 'claude-2' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e.g. "claude-2" or "claude-2-100k".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-2' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-2.v1")
def anthropic_claude_2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["claude-2", "claude-2-100k"] = "claude-2",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-2' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-2", "claude-2-100k"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-1.v2")
def anthropic_claude_1_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-1",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e. g. "claude-1" or "claude-1-100k".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-1.v1")
def anthropic_claude_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["claude-1", "claude-1-100k"] = "claude-1",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-1", "claude-1-100k"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-instant-1.v2")
def anthropic_claude_instant_1_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-instant-1",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-instant-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e. g. "claude-instant-1" or "claude-instant-1-100k".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-instant-1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-instant-1.v1")
def anthropic_claude_instant_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "claude-instant-1", "claude-instant-1-100k"
    ] = "claude-instant-1",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-instant-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-instant-1", "claude-instant-1-100k"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-instant-1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-instant-1-1.v2")
def anthropic_claude_instant_1_1_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-instant-1.1",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-instant-1.1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e. g. "claude-instant-1.1" or "claude-instant-1.1-100k".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-instant-1.1'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-instant-1-1.v1")
def anthropic_claude_instant_1_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "claude-instant-1.1", "claude-instant-1.1-100k"
    ] = "claude-instant-1.1",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-instant-1.1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-instant-1.1", "claude-instant-1.1-100k"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-instant-1.1' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-1-0.v2")
def anthropic_claude_1_0_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-1.0",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.0' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e. g. "claude-1.0".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-1.0'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-1-0.v1")
def anthropic_claude_1_0(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["claude-1.0"] = "claude-1.0",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.0' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-1.0"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-1.0' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-1-2.v2")
def anthropic_claude_1_2_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-1.2",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.2' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model to use, e. g. "claude-1.2".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-1.2'.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-1-2.v1")
def anthropic_claude_1_2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["claude-1.2"] = "claude-1.2",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.2' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-1.2"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-1.2' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )


@registry.llm_models("spacy.Claude-1-3.v2")
def anthropic_claude_1_3_v2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "claude-1.3",
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.3' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (str): Name of model variant to use, e. g. "claude-1.3" or "claude-1.3-100k".
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
    RETURNS (Anthropic): Anthropic instance for 'claude-1.3' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Claude-1-3.v1")
def anthropic_claude_1_3(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["claude-1.3", "claude-1.3-100k"] = "claude-1.3",  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Anthropic instance for 'claude-1.3' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["claude-1.3", "claude-1.3-100k"]): Model variant to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Anthropic): Anthropic instance for 'claude-1.3' model.
    """
    return Anthropic(
        name=name,
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=None,
    )
