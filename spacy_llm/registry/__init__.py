import spacy

# Create new registry.
if "llm" not in spacy.registry.get_registry_names():
    spacy.registry.create("llm", entry_points=True)

from .task import noop_task
from .api import api_minichain
from .prompt import minichain_simple_prompt, langchain_simple_prompt

__all__ = [
    # task
    "noop_task",
    # api
    "api_minichain",
    # prompt
    "minichain_simple_prompt",
    "langchain_simple_prompt",
]
