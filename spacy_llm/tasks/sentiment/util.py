from typing import Any, Dict, Iterable

from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample


class SentimentExample(FewshotExample):
    text: str
    score: float

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        return cls(
            text=example.reference.text,
            score=getattr(example.reference._, kwargs["field"]),
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
