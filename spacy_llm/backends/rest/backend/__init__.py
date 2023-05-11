from . import base, openai, noop

supported_apis = {
    "OpenAI": openai.OpenAIBackend,
    "NoOp": noop.NoOpBackend,
}

__all__ = ["base", "openai", "noop", "supported_apis"]
