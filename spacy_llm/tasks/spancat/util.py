from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ..span import SpanExample
from ..span.examples import SpanCoTExample
from .task import SpanCatTask


class SpanCatExample(SpanExample[SpanCatTask]):
    @classmethod
    def generate(cls, example: Example, task: SpanCatTask) -> Optional[Self]:
        entities = defaultdict(list)
        for span in example.reference.spans[task.spans_key]:
            entities[span.label_].append(span.text)

        return cls(text=example.reference.text, entities=entities)


class SpanCatCoTExample(SpanCoTExample[SpanCatTask]):
    @classmethod
    def generate(cls, example: Example, task: SpanCatTask) -> Self:
        return cls(
            text=example.reference.text,
            spans=SpanCoTExample._extract_span_reasons(
                example.reference.spans[task.spans_key]
            ),
        )


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


def reduce_shards_to_doc(task: SpanCatTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for SpanCatTask.
    task (SpanCatTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # SpanCatTask only affects span-specific information, so we can just merge shards.
    return Doc.from_docs(list(shards), ensure_whitespace=True)
