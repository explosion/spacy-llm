import catalogue
import confection
import spacy


class registry(confection.registry):
    @classmethod
    def init(cls):
        for registry_name in ("prompts", "apis", "tasks"):
            if f"llm_{registry_name}" not in spacy.registry.get_registry_names():
                new_registry = catalogue.create(
                    "spacy-llm", registry_name, entry_points=True
                )
                setattr(spacy.registry, f"llm_{registry_name}", new_registry)
                setattr(cls, registry_name, new_registry)


registry.init()
