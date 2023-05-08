try:
    import langchain
except (ImportError, AttributeError):
    langchain = None

try:
    import minichain
except (ImportError, AttributeError):
    minichain = None

try:
    import torch

    has_torch = True
except ImportError:
    torch = None
    has_torch = False

try:
    import transformers

    has_transformers = True
except ImportError:
    transformers = None
    has_transformers = False

has = {"minichain": minichain is not None, "langchain": langchain is not None}
