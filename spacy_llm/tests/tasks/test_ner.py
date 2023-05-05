import spacy
from confection import Config

cfg_string = """
[nlp]
lang = "es"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm] 
factory = "llm"

[components.llm.task]
@llm_tasks: "spacy.NERZeroShot.v1"
labels: ["PER", "ORG", "LOC"]}

[components.llm.backend]
@llm_backends: "spacy.MiniChain.v1"
api: "OpenAI"
config: {}
"""


def test_ner_config():
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


def test_ner_io():
    pass
