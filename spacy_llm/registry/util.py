import spacy

for registry_name in ("queries", "backends", "tasks", "misc"):
    if f"llm_{registry_name}" not in spacy.registry.get_registry_names():
        spacy.registry.create(f"llm_{registry_name}", entry_points=True)
