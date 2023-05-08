from typing import Callable, Iterable, Dict, Any

import spacy
from spacy.util import SimpleFrozenDict
from .backend import Backend


@spacy.registry.llm_queries("spacy.CallREST.v1")
def query_rest() -> Callable[[Backend, Iterable[str]], Iterable[str]]:
    """Returns query Callable for minimal REST backend.
    RETURNS (Callable[[Backend, Iterable[str]], Iterable[str]]:): Callable executing simple prompts using a Backend
        instance.
    """

    def prompt(backend: Backend, prompts: Iterable[str]) -> Iterable[str]:
        return backend(prompts)

    return prompt


@spacy.registry.llm_backends("spacy.REST.v1")
def backend_rest(
    api: str,
    strict: bool,
    query: Callable[[Backend, Iterable[str]], Iterable[str]] = query_rest(),
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable using minimal REST backend to prompt specified API.
    api (str): Name of any API. Currently supported: "OpenAI".
    query (Callable[[Backend, Iterable[str]], Iterable[str]]): Callable implementing querying this API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the Backend instance.
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable using the querying the specified API using a
        Backend instance.
    """

    backend = Backend(api=api, config=config, strict=strict)

    def _query(prompts: Iterable[str]) -> Iterable[str]:
        return query(backend, prompts)

    return _query
