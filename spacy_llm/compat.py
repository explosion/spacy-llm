import sys

try:
    import langchain
except (ImportError, AttributeError):
    langchain = None

try:
    import minichain
except (ImportError, AttributeError):
    minichain = None

if sys.version_info[:2] >= (3, 8):  # Python 3.8+
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable  # noqa: F401

has = {"minichain": minichain is not None, "langchain": langchain is not None}
