from typing import Any, Dict, Iterable, Optional

from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import SentimentTask


class SentimentExample(FewshotExample[SentimentTask]):
    text: str
    score: float

    @classmethod
    def generate(cls, example: Example, task: SentimentTask) -> Optional[Self]:
        return cls(
            text=example.reference.text,
            score=getattr(example.reference._, task.field),
        )


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score sentiment accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    score_diffs = [
        abs(
            getattr(example.predicted._, kwargs["field"])
            - getattr(example.reference._, kwargs["field"])
        )
        for example in examples
    ]

    return {"acc_sentiment": 1 - (sum(score_diffs) / len(score_diffs))}
