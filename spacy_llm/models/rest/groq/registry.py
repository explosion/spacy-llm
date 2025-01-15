from typing import Any, Dict, Optional

from confection import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Endpoints, Groq

_DEFAULT_TEMPERATURE = 0.0

"""
Parameter explanations:
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    endpoint (Optional[str]): Endpoint to set. Defaults to standard endpoint.
"""


@registry.llm_models("spacy.groq.v1")
def groq(
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    name: str = "llama-3.1-70b-versatile",
    strict: bool = Groq.DEFAULT_STRICT,
    max_tries: int = Groq.DEFAULT_MAX_TRIES,
    interval: float = Groq.DEFAULT_INTERVAL,
    max_request_time: float = Groq.DEFAULT_MAX_REQUEST_TIME,
    endpoint: Optional[str] = None,
    context_length: Optional[int] = None,
) -> Groq:
    """Returns Groq instance for 'llama-3.1-70b-versatile' model using REST to prompt API.

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    name (str): Model name to use. Can be any model name supported by the Groq API - e. g. 'llama-3.1-70b-versatile',
        "llama-3.1-70b-versatile", ....
    context_length (Optional[int]): Context length for this model. Only necessary for sharding and if no context length
        natively provided by spacy-llm.
    RETURNS (Groq): Groq instance for 'llama-3.1-70b-versatile' model.

    DOCS: https://spacy.io/api/large-language-models#models
    """
    return Groq(
        name=name,
        endpoint=endpoint or Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )
