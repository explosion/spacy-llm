import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.tasks.ner import find_substrings, ner_zeroshot_task

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

[components.llm.task.normalizer]
@misc: "spacy.UppercaseNormalizer.v1"

[components.llm.backend]
@llm_backends: "spacy.MiniChain.v1"
api: "OpenAI"
config: {}
"""


def test_ner_config():
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


@pytest.mark.external
def test_ner_predict():
    """Use OpenAI to get zero-shot NER results.

    Issues with this test:
     - behaviour is unguaranteed to be consistent/predictable
     - on every run, a cost is occurred with OpenAI.
    """
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)
    assert len(doc.ents) > 0
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.external
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


@pytest.mark.parametrize(
    "text,substrings",
    [
        (
            "Felipe and Jaime went to the library.",
            ["Felipe", "Jaime", "library"],
        ),  # simple
        (
            "The Manila Observatory was founded in 1865.",
            ["Manila", "The Manila Observatory"],
        ),  # overlapping
        (
            "Take the road from Downtown and turn left at the public market.",
            ["public market", "Downtown"],
            # flipped
        ),
    ],
)
def test_ensure_offsets_correspond_to_substrings(text, substrings):
    offsets = find_substrings(text, substrings)
    # Compare strings instead of offsets, but we need to get
    # those strings first from the text
    found_substrings = [text[start:end] for start, end in offsets]
    assert substrings == found_substrings


@pytest.mark.parametrize(
    "text,response,gold_ents",
    [
        # simple
        (
            "Felipe and Jaime went to the library.",
            "PER: Felipe, Jaime\nLOC: library",
            [("Felipe", "PER"), ("Jaime", "PER"), ("library", "LOC")],
        ),
        # overlapping: should only return the longest span
        (
            "The Manila Observatory was founded in 1865.",
            "LOC: The Manila Observatory, Manila, Manila Observatory",
            [("The Manila Observatory", "LOC")],
        ),
        # flipped: order shouldn't matter
        (
            "Take the road from Downtown and turn left at the public market.",
            "LOC: public market, Downtown",
            [("Downtown", "LOC"), ("public market", "LOC")],
        ),
    ],
)
def test_ner_zero_shot_task(text, response, gold_ents):
    labels = "PER,ORG,LOC"
    _, parser = ner_zeroshot_task(labels=labels)
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp(text)
    # Pass to the parser
    # Note: parser() returns a list so we get what's inside
    doc_out = list(parser([doc_in], [response]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents
