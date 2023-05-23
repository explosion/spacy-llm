import spacy
from spacy.tokens import Span

from spacy_llm.tasks.rel import _preannotate


def test_text_preannotation():
    nlp = spacy.load("blank:en")

    doc = nlp("This is a test")
    doc.ents = [Span(doc, start=3, end=4, label="test")]

    assert _preannotate(doc) == "This is a test[ENT0:test]"
