from typing import Any, Dict, Iterable

from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc
from spacy.training import Example

from .task import SpanCatTask


def score(
    task: SpanCatTask, docs: Iterable[Doc], examples: Iterable[Example]
) -> Dict[str, Any]:
    """Score spancat accuracy in examples.
    task (SpanCatTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return spancat_score(
        examples,
        spans_key=task.spans_key,
        allow_overlap=True,
    )
