from spacy.training import Example

from spacy_llm.ty import FewshotExample


class SentimentExample(FewshotExample):
    text: str
    score: float

    @classmethod
    def generate(cls, example: Example, **kwargs) -> "SentimentExample":
        return SentimentExample(
            text=example.reference.text,
            score=getattr(example.reference._, kwargs["field"]),
        )
