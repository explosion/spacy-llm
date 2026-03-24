from .hf import dolly_hf, openllama_hf, stablelm_hf
from .rest import anthropic, cohere, noop, openai, palm

__all__ = [
    "anthropic",
    "cohere",
    "openai",
    "dolly_hf",
    "noop",
    "stablelm_hf",
    "openllama_hf",
    "palm",
]


def _register_langchain_models():
    """Lazily import and register langchain models to avoid importing
    langchain at module level (it may not be installed, or may be
    incompatible with the current Python version)."""
    try:
        from .langchain import LangChain

        LangChain.register_models()
    except (ImportError, Exception):
        pass
