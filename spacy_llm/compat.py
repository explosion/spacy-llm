try:
    import langchain
except (ImportError, AttributeError):
    langchain = None

try:
    import minichain
except (ImportError, AttributeError):
    minichain = None

has = {"minichain": minichain is not None, "langchain": langchain is not None}
