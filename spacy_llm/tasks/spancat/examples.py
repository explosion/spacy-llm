from collections import defaultdict

from spacy.training import Example

from ...compat import Self
from ..span import SpanExample


class SpanCatExample(SpanExample):
    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        entities = defaultdict(list)
        for span in example.reference.spans[kwargs["spans_key"]]:
            entities[span.label_].append(span.text)

        return cls(text=example.reference.text, entities=entities)
