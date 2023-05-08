from typing import Callable, Iterable, Dict, Any

import spacy
from spacy.util import SimpleFrozenDict
from .backend import Backend


@spacy.registry.llm_queries("spacy.CallMinimal.v1")
def query_minimal() -> Callable[[Backend, Iterable[str]], Iterable[str]]:
    """Returns query Callable for minimal backend.
    RETURNS (Callable[[Backend, Iterable[str]], Iterable[str]]:): Callable executing simple prompts using a Backend
        instance.
    """

    def prompt(backend: Backend, prompts: Iterable[str]) -> Iterable[str]:
        return backend(prompts)

    return prompt


@spacy.registry.llm_backends("spacy.Minimal.v1")
def backend_minimal(
    api: str,
    query: Callable[[Backend, Iterable[str]], Iterable[str]] = query_minimal(),
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable using minimal backend to prompt specified API.
    api (str): Name of any API. Currently supported: "OpenAI".
    query (Callable[[Backend, Iterable[str]], Iterable[str]]): Callable implementing querying this API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the Backend instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable using the querying the specified API using a Backend
        instance.
    """

    backend = Backend(api=api, config=config)

    def _query(prompts: Iterable[str]) -> Iterable[str]:
        return query(backend, prompts)

    return _query
