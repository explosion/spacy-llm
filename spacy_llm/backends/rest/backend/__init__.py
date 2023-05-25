from typing import Type, Dict

from . import base, openai, noop

supported_apis: Dict[str, Type[base.Backend]] = {
    "OpenAI": openai.OpenAIBackend,
    "NoOp": noop.NoOpBackend,
}

__all__ = ["base", "openai", "noop", "supported_apis"]
