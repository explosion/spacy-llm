# mypy: ignore-errors
import sys

if sys.version_info[:2] >= (3, 8):  # Python 3.8+
    from typing import Literal, Protocol, runtime_checkable
else:
    from typing_extensions import runtime_checkable  # noqa: F401
    from typing_extensions import Literal, Protocol  # noqa: F401

if sys.version_info[:2] >= (3, 9):  # Python 3.9+
    from typing import TypedDict  # noqa: F401
else:
    from typing_extensions import TypedDict  # noqa: F401

if sys.version_info[:2] >= (3, 11):  # Python 3.11+
    from typing import Self  # noqa: F401
else:
    from typing_extensions import Self  # noqa: F401

try:
    import langchain
    import langchain_community

    has_langchain = True
except (ImportError, AttributeError):
    langchain = None
    langchain_community = None
    has_langchain = False

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


from pydantic import BaseModel  # noqa: F401
from pydantic import ConfigDict  # noqa: F401
from pydantic import ValidationError  # noqa: F401
from pydantic import field_validator  # noqa: F401
