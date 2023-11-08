from typing import Iterable, Optional

from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import SummarizationTask


class SummarizationExample(FewshotExample[SummarizationTask]):
    text: str
    summary: str

    @classmethod
    def generate(cls, example: Example, task: SummarizationTask) -> Optional[Self]:
        return cls(
            text=example.reference.text,
            summary=getattr(example.reference._, task.field),
        )


def reduce_shards_to_doc(shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for SummarizationTask.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # Summaries are per shard, so we can merge. Number of shards is considered in max. number of words. This means that
    # the resulting summaries will be per shard, which should be an approximately correct summary still.
    return Doc.from_docs(list(shards), ensure_whitespace=True)
