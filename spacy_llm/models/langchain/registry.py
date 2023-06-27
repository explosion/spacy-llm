from ...compat import has_langchain
from ...registry import registry
from . import LangChain

# Dynamically register LangChain API classes as individual models, if this hasn't been done yet.
if has_langchain and not any(
    [("langchain" in handle) for handle in registry.llm_models.get_all()]
):
    LangChain.register_models()
