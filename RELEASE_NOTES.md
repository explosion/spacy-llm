Python 3.14 support, Pydantic v2 migration, and updated dependencies

## Features

- Add support for Python 3.13 and 3.14 (Python 3.9 dropped; minimum is now 3.10)
- Migrate from Pydantic v1 to Pydantic v2 native API
- Update LangChain integration to require langchain >= 1.0

## Fixes

- Fix Jinja2 template sandbox escaping issue (#491)
- Fix compatibility with spaCy 3.8.13 by requiring confection >= 1.3.3
- Filter spurious pydantic v1 deprecation warning from langchain-core on Python 3.14

## Other

- Update LangChain import to be lazy-loaded for Python 3.14 compatibility
- Update dev dependencies: langchain 1.x, openai 1.x, drop stale pins
