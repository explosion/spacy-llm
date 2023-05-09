try:
    import langchain

    has_langchain = True
except (ImportError, AttributeError):
    langchain = None  # type: ignore[assignment]
    has_langchain = False

try:
    import minichain  # type: ignore[import]

    has_minichain = True
except (ImportError, AttributeError):
    minichain = None
    has_minichain = False

try:
    import torch  # type: ignore[import]

    has_torch = True
except ImportError:
    torch = None
    has_torch = False

try:
    import transformers  # type: ignore[import]

    has_transformers = True
except ImportError:
    transformers = None
    has_transformers = False

try:
    import accelerate  # type: ignore[import]

    has_accelerate = True
except ImportError:
    accelerate = None
    has_accelerate = False
