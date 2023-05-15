import os

if os.getenv("OPENAI_API_KEY") is None:
    has_openai_key = False
else:
    has_openai_key = True
