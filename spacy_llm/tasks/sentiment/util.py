from typing import Optional

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
