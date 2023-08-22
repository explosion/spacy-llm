from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from spacy.scorer import get_ner_prf
from spacy.training import Example

from ...compat import Self
from ..span import SpanExample


class NERExample(SpanExample):
    @classmethod
    def generate(cls, example: Example, **kwargs) -> Optional[Self]:
        entities = defaultdict(list)
        for ent in example.reference.ents:
            entities[ent.label_].append(ent.text)

        return cls(text=example.reference.text, entities=entities)


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score NER accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return get_ner_prf(examples)
