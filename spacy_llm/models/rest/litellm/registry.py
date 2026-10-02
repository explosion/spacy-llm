from typing import Any, Dict, Optional

from confection import SimpleFrozenDict

from ....registry import registry
from .model import LiteLLM

_DEFAULT_TEMPERATURE = 0.0


@registry.llm_models("spacy.LiteLLM.v1")
def litellm_v1(
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    name: str = "openai/gpt-4o",
    strict: bool = LiteLLM.DEFAULT_STRICT,
    max_tries: int = LiteLLM.DEFAULT_MAX_TRIES,
    interval: float = LiteLLM.DEFAULT_INTERVAL,
    max_request_time: float = LiteLLM.DEFAULT_MAX_REQUEST_TIME,
    endpoint: Optional[str] = None,
    context_length: Optional[int] = None,
) -> LiteLLM:
    """Returns LiteLLM instance for any model supported by LiteLLM (100+ providers).

    Uses the LiteLLM Python SDK to route requests to any LLM provider.
    Model names follow the litellm format: provider/model-name
    (e.g. "anthropic/claude-haiku-4-5", "openai/gpt-4o", "bedrock/anthropic.claude-3-haiku").

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    name (str): Model name in litellm format (provider/model-name).
    context_length (Optional[int]): Context length for this model.
    RETURNS (LiteLLM): LiteLLM instance.
    """
    return LiteLLM(
        name=name,
        endpoint=endpoint or "",
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )
