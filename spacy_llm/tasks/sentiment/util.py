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
