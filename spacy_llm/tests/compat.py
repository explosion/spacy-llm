import os

has_openai_key = os.getenv("OPENAI_API_KEY") is not None
has_anthropic_key = os.getenv("ANTHROPIC_API_KEY") is not None
has_cohere_key = os.getenv("CO_API_KEY") is not None
has_azure_openai_key = os.getenv("AZURE_OPENAI_KEY") is not None
has_palm_key = os.getenv("PALM_API_KEY") is not None
