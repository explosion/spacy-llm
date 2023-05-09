# mypy: ignore-errors
import sys

try:
    # Ensure we import langchain only in the supported Python versions.
    if sys.version_info[1] not in (9, 10, 11):
        raise ImportError
    import langchain

    has_langchain = True
except (ImportError, AttributeError):
    langchain = None
    has_langchain = False

try:
    # Ensure we import minichain only in the supported Python versions.
    if sys.version_info[1] not in (7, 8, 9):
        raise ImportError
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
