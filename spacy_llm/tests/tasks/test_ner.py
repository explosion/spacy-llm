import spacy
from confection import Config
from spacy.util import make_tempdir

cfg_string = """
[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm] 
factory = "llm"

[components.llm.task]
@llm_tasks: "spacy.NERZeroShot.v1"
labels: PER,ORG,LOC

[components.llm.backend]
@llm_backends: "spacy.MiniChain.v1"
api: "OpenAI"
config: {}
"""


def test_ner_config():
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


def test_ner_predict():
    """Use OpenAI to get zero-shot NER results.

    Issues with this test:
     - behaviour is ungaranteed to be consistent/predictable
     - on every run, a cost is occurred with OpenAI.
    """
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)
    assert len(doc.ents) > 0
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]


def test_ner_io():
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    text = "Marc and Bob both live in Ireland."
    doc = nlp2(text)
    assert len(doc.ents) > 0
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]
