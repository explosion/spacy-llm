from spacy.training import Example

from ...compat import BaseModel, Self


class SummarizationExample(BaseModel):
    text: str
    summary: str

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        return cls(
            text=example.reference.text,
            summary=getattr(example.reference._, kwargs["field"]),
        )
