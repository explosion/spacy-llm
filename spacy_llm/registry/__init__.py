import spacy

# Create new registry.
if "llm" not in spacy.registry.get_registry_names():
    spacy.registry.create("llm", entry_points=True)

from .template import noop_template
from .api import api_minichain
from .prompt import minichain_simple_prompt, langchain_simple_prompt
from .parse import noop_parse

__all__ = [
    # template
    "noop_template",
    # api
    "api_minichain",
    # prompt
    "minichain_simple_prompt",
    "langchain_simple_prompt",
    # parse
    "noop_parse",
]
