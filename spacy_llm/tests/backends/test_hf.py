import spacy
from confection import Config

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.HF.v1",
        "model": "databricks/dolly-v2-3b",
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}

NLP_CONFIG = """

[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm] 
factory = "llm"

[components.llm.task]
@llm_tasks: "spacy.NoOp.v1"

[components.llm.backend]
@llm_backends: "spacy.HF.v1"
model: "databricks/dolly-v2-3b"
"""


def test_init():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


def test_init_from_config():
    orig_config = Config().from_str(NLP_CONFIG)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
