from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import SummarizationTask


class SummarizationExample(FewshotExample[SummarizationTask]):
    text: str
    summary: str

    @classmethod
    def generate(cls, example: Example, task: SummarizationTask) -> Self:
        return cls(
            text=example.reference.text,
            summary=getattr(example.reference._, task.field),
        )
