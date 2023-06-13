import os

if os.getenv("OPENAI_API_KEY") is None:
    has_openai_key = False
else:
    has_openai_key = True

if os.getenv("ANTHROPIC_API_KEY") is None:
    has_anthropic_key = False
else:
    has_anthropic_key = True

if os.getenv("CO_API_KEY") is None:
    has_cohere_key = False
else:
    has_cohere_key = True
