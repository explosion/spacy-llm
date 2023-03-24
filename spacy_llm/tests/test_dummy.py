import spacy


def test_init():
    nlp = spacy.load("blank:en")
    assert nlp
    # todo configure generator functions
