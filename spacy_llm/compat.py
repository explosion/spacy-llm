try:
    import langchain

    has_langchain = True
except (ImportError, AttributeError):
    langchain = None
    has_langchain = False

try:
    import minichain

    has_minichain = True
except (ImportError, AttributeError):
    minichain = None
    has_minichain = False

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

try:
    import accelerate

    has_accelerate = True
except ImportError:
    accelerate = None
    has_accelerate = False

has = {"minichain": minichain is not None, "langchain": langchain is not None}
