from spacy_llm.tasks.rel.items import EntityItem
from spacy_llm.tasks.rel.util import RELExample


def test_to_doc_maps_entity_offsets_after_punctuation_normalization():
    example = RELExample(
        text="Alice, Bob works.",
        ents=[EntityItem(start_char=7, end_char=10, label="PERSON")],
        relations=[],
    )

    doc = example.to_doc()

    assert [(ent.text, ent.label_) for ent in doc.ents] == [("Bob", "PERSON")]
