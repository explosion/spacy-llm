import warnings
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


def reduce_shards_to_doc(task: SummarizationTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for SummarizationTask.
    task (SummarizationTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    shards = list(shards)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping .* while merging docs.",
        )
        doc = Doc.from_docs(list(shards), ensure_whitespace=True)

    # Summaries are per shard, so we can merge. Number of shards is considered in max. number of words. This means that
    # the resulting summaries will be per shard, which should be an approximately correct summary still.
    setattr(
        doc._, task.field, " ".join([getattr(shard._, task.field) for shard in shards])
    )

    return doc
