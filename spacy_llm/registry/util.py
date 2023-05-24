import catalogue  # type: ignore[import]
import spacy

for registry_name in ("queries", "backends", "tasks", "misc"):
    if f"llm_{registry_name}" not in spacy.util.registry.get_registry_names():
        spacy.util.registry.create(f"llm_{registry_name}", entry_points=True)


class registry(spacy.util.registry):
    llm_backends: catalogue.Registry
    llm_queries: catalogue.Registry
    llm_tasks: catalogue.Registry
    llm_misc: catalogue.Registry
