from ...compat import has_langchain
from ...registry import registry
from .model import LangChain, query_langchain

# Dynamically register LangChain API classes as individual models, if this hasn't been done yet.
if has_langchain and not any(
    [("langchain" in handle) for handle in registry.llm_models.get_all()]
):
    LangChain.register_models()

__all__ = [
    "LangChain",
    "query_langchain",
]
