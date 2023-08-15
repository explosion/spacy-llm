from typing import Any, Dict, Iterable

from spacy.pipeline.spancat import spancat_score
from spacy.training import Example


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
