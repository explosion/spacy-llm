from typing import Any, Callable, Dict, Iterable

from spacy.util import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Endpoints, OpenAI


@registry.llm_models("spacy.GPT-4.v1")
def openai_gpt_4(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"
    ] = "gpt-4",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'gpt-4' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["0314", "32k", "32k-0314"]]): Model to use. Base 'gpt-4' model by default.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'gpt-4' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.GPT-3-5.v1")
def openai_gpt_3_5(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-0613-16k",
    ] = "gpt-3.5-turbo",  # noqa: F722,F821
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'gpt-3.5' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Literal["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-0613-16k"]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'gpt-3.5' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Text-Davinci.v1")
def openai_text_davinci(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal[
        "text-davinci-002", "text-davinci-003"
    ] = "text-davinci-003",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-davinci' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["text-davinci-002", "text-davinci-003"]]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'text-davinci' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Code-Davinci.v1")
def openai_code_davinci(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["code-davinci-002"] = "code-davinci-002",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'code-davinci' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["code-davinci-002"]]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'code-davinci' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Text-Curie.v1")
def openai_text_curie(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["text-curie-001"] = "text-curie-001",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-curie' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["text-curie-001"]]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'text-curie' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Text-Babbage.v1")
def openai_text_babbage(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["text-babbage-001"] = "text-babbage-001",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-babbage' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["text-babbage-001"]]): Model to use.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): OpenAI instance for 'text-babbage' model using REST to
        prompt API.
    """
    return OpenAI(
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Text-Ada.v1")
def openai_text_ada(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["text-ada-001"] = "text-ada-001",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'text-ada' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["text-ada-001"]]): Model to use.
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
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Davinci.v1")
def openai_davinci(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["davinci"] = "davinci",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'davinci' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["davinci"]]): Model to use.
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
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Curie.v1")
def openai_curie(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["curie"] = "curie",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'curie' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["curie"]]): Model to use.
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
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Babbage.v1")
def openai_babbage(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["babbage"] = "babbage",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'babbage' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["babbage"]]): Model to use.
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
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )


@registry.llm_models("spacy.Ada.v1")
def openai_ada(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: Literal["ada"] = "ada",  # noqa: F722
    strict: bool = OpenAI.DEFAULT_STRICT,
    max_tries: int = OpenAI.DEFAULT_MAX_TRIES,
    interval: float = OpenAI.DEFAULT_INTERVAL,
    max_request_time: float = OpenAI.DEFAULT_MAX_REQUEST_TIME,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns OpenAI instance for 'ada' model using REST to prompt API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the model instance.
    name (Optional[Literal["ada"]]): Model to use.
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
        name=name,
        endpoint=Endpoints.NON_CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )
