from typing import Any, Dict

from confection import SimpleFrozenDict

from ....registry import registry
from .model import Endpoints, Ollama

@registry.llm_models("spacy.Ollama.v1")
def ollama_mistral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mistral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096
) -> Ollama:
    """Returns Ollama instance for 'mistral' model.
    
    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    name (str): Model name to use. Defaults to 'mistral'. 
    strict (bool): Whether to raise exception on API errors. Defaults to Ollama.DEFAULT_STRICT.
    max_tries (int): Max number of API request retries. Defaults to Ollama.DEFAULT_MAX_TRIES. 
    interval (float): Retry interval in seconds. Defaults to Ollama.DEFAULT_INTERVAL.
    max_request_time (float): Max API request time in seconds. Defaults to Ollama.DEFAULT_MAX_REQUEST_TIME.
    context_length (int): Max context length. Defaults to 4096.

    RETURNS (Ollama): Ollama instance for 'mistral' model
    """
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length
    )
