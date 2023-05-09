from .dolly import backend_dolly_hf
from .langchain import backend_langchain, query_langchain
from .minichain import backend_minichain, query_minichain
from .rest import backend_rest

from dotenv import load_dotenv  # type: ignore[import]

load_dotenv()  # take environment variables from .env.

__all__ = [
    "query_minichain",
    "query_langchain",
    "backend_minichain",
    "backend_langchain",
    "backend_rest",
    "backend_dolly_hf",
]
