from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from spacy.scorer import get_ner_prf
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ..span import SpanExample
from ..span.examples import SpanCoTExample
from .task import NERTask


class NERExample(SpanExample[NERTask]):
    @classmethod
    def generate(cls, example: Example, task: NERTask) -> Optional[Self]:
        entities = defaultdict(list)
        for ent in example.reference.ents:
            entities[ent.label_].append(ent.text)

        return cls(text=example.reference.text, entities=entities)


class NERCoTExample(SpanCoTExample[NERTask]):
    @classmethod
    def generate(cls, example: Example, task: NERTask) -> Optional[Self]:
        return cls(
            text=example.reference.text,
            spans=SpanCoTExample._extract_span_reasons(example.reference.ents),
        )


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score NER accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return get_ner_prf(examples)


def reduce_shards_to_doc(shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for NERTask.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # todo this is yet a dummy implementation that will only return the first doc shard.
    return list(shards)[0]
