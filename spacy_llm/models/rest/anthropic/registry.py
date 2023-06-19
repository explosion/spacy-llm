from typing import Any, Callable, Dict, Iterable, Optional

from spacy.util import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Anthropic, Endpoints


@registry.llm_models("spacy.claude-1.Anthropic.v1")
def anthropic_claude_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Optional[Literal["100k"]] = None,  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["100k"]]): Model variant to use. Base 'claude-1' model by default.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-1' model using REST to
        prompt API.
    """
    return Anthropic(
        name=f"claude-1{('-' + variant) if variant else ''}",
        endpoint=Endpoints.COMPLETIONS,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.claude-instant-1.Anthropic.v1")
def anthropic_claude_instant_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Optional[Literal["100k"]] = None,  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-instant-1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["100k"]]): Model variant to use. Base 'claude-instant-1' model by default.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-instant-1' model using REST to
        prompt API.
    """
    return Anthropic(
        name=f"claude-instant-1{('-' + variant) if variant else ''}",
        endpoint=Endpoints.COMPLETIONS.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.claude-instant-1.1.Anthropic.v1")
def anthropic_claude_instant_1_1(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Optional[Literal["100k"]] = None,  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-instant-1.1' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["100k"]]): Model variant to use. Base 'claude-instant-1.1' model by default.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-instant-1.1' model using REST to
        prompt API.
    """
    return Anthropic(
        name=f"claude-instant-1{('-' + variant) if variant else ''}",
        endpoint=Endpoints.COMPLETIONS,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.claude-1.0.Anthropic.v1")
def anthropic_claude_1_0(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-1.0' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-1.0' model using REST to prompt
        API.
    """
    return Anthropic(
        name="claude-1.0",
        endpoint=Endpoints.COMPLETIONS,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.claude-1.2.Anthropic.v1")
def anthropic_claude_1_2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-1.2' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-1.2' model using REST to prompt
        API.
    """
    return Anthropic(
        name="claude-1.2",
        endpoint=Endpoints.COMPLETIONS,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.claude-1.3.Anthropic.v1")
def anthropic_claude_1_3(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Optional[Literal["100k"]] = None,  # noqa: F722
    strict: bool = Anthropic.DEFAULT_STRICT,
    max_tries: int = Anthropic.DEFAULT_MAX_TRIES,
    interval: float = Anthropic.DEFAULT_INTERVAL,
    max_request_time: float = Anthropic.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Anthropic instance for 'claude-1.3' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["100k"]]): Model variant to use. Base 'claude-1.3' model by default.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Anthropic instance for 'claude-1.3' model using REST to prompt
        API.
    """
    return Anthropic(
        name=f"claude-1.3{('-' + variant) if variant else ''}",
        endpoint=Endpoints.COMPLETIONS,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )
