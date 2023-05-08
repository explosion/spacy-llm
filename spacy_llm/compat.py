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
