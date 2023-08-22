from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from spacy.pipeline.spancat import spancat_score
from spacy.training import Example

from ...compat import Self
from ..span import SpanExample


class SpanCatExample(SpanExample):
    @classmethod
    def generate(cls, example: Example, **kwargs) -> Optional[Self]:
        entities = defaultdict(list)
        for span in example.reference.spans[kwargs["spans_key"]]:
            entities[span.label_].append(span.text)

        return cls(text=example.reference.text, entities=entities)


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score spancat accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return spancat_score(
        examples,
        spans_key=kwargs["spans_key"],
        allow_overlap=True,
    )
