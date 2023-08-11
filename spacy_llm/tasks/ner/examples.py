from collections import defaultdict

from spacy.training import Example

from ...compat import Self
from ..span import SpanExample


class NERExample(SpanExample):
    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        entities = defaultdict(list)
        for ent in example.reference.ents:
            entities[ent.label_].append(ent.text)

        return cls(text=example.reference.text, entities=entities)
