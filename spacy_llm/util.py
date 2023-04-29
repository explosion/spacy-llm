import catalogue
import spacy


class registry(spacy.registry):
    tasks = catalogue.create("spacy-llm", "tasks", entry_points=True)
    apis = catalogue.create("spacy-llm", "apis", entry_points=True)
    prompts = catalogue.create("spacy-llm", "prompts", entry_points=True)
