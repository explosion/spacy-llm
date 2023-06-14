from typing import Any, Callable, Dict, Iterable, Type


from spacy.util import SimpleFrozenDict

from . import anthropic, base, openai, noop
from ...registry import registry


supported_apis: Dict[str, Type[base.Backend]] = {
    "OpenAI": openai.OpenAIBackend,
    "Anthropic": anthropic.AnthropicBackend,
    "NoOp": noop.NoOpBackend,
}


@registry.llm_backends("spacy.REST.v1")
def backend_rest(
    api: str,
    config: Dict[Any, Any] = SimpleFrozenDict(),
    strict: bool = True,
    max_tries: int = 5,
    interval: float = 1.0,
    max_request_time: float = 30,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable using minimal REST backend to prompt specified API.
    api (str): Name of any API. Currently supported: "OpenAI".
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the Backend instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable using the querying the specified API using a
        Backend instance.
    """

    backend = supported_apis[api](
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
    )
    return backend
