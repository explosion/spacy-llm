# mypy: ignore-errors
import sys
import warnings

if sys.version_info[:2] >= (3, 8):  # Python 3.8+
    from typing import Protocol, runtime_checkable, Literal
else:
    from typing_extensions import Protocol, runtime_checkable, Literal  # noqa: F401

try:
    import langchain

    has_langchain = True
except (ImportError, AttributeError):
    langchain = None
    has_langchain = False

try:
    with warnings.catch_warnings(category=DeprecationWarning):
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
