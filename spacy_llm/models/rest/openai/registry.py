from typing import Any, Callable, Dict, Iterable, Optional

from spacy.util import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Endpoints, OpenAI


@registry.llm_models("spacy.gpt-4.OpenAI.v1")
def openai_gpt_4(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Optional[Literal["0314", "32k", "32k-0314"]] = None,  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'gpt-4' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["0314", "32k", "32k-0314"]]): Model variant to use. Base 'gpt-4' model by default.
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
    return OpenAI(
        name=f"gpt-4{('-' + variant) if variant else ''}",
        endpoint=Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.gpt-3.5.OpenAI.v1")
def openai_gpt_3_5(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Literal["turbo", "turbo-0301"] = "turbo",  # noqa: F722,F821
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'gpt-3.5' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Literal["turbo", "turbo-0301"]): Model variant to use. 'turbo' by default.
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
    return OpenAI(
        name=f"gpt-3.5{('-' + variant) if variant else ''}",
        endpoint=Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.text-davinci.OpenAI.v1")
def openai_text_davinci(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    variant: Literal["002", "003"] = "003",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-davinci' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    variant (Optional[Literal["002", "003"]]): Model variant to use. '003' by default.
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
    return OpenAI(
        name=f"text-davinci{('-' + variant) if variant else ''}",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.text-curie.OpenAI.v1")
def openai_text_curie(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-curie' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="text-curie-001",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.text-babbage.OpenAI.v1")
def openai_text_babbage(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-babbage' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="text-babbage-001",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.text-ada.OpenAI.v1")
def openai_text_ada(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-ada' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="text-ada-001",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.davinci.OpenAI.v1")
def openai_davinci(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'davinci' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="davinci",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.curie.OpenAI.v1")
def openai_curie(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'curie' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="curie",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.babbage.OpenAI.v1")
def openai_babbage(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'babbage' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="babbage",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.ada.OpenAI.v1")
def openai_ada(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'ada' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
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
    return OpenAI(
        name="ada",
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )
