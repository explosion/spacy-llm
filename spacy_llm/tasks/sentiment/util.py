from typing import Iterable, Optional

from spacy.tokens import Doc
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


def reduce_shards_to_doc(shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for SentimentTask.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # todo make generic, pass task to shard reducers (necessary e. g. for tasks with dynamic fields in docs)
    # todo Weight-average sentiment scores over shards based on token count per shard.
    return list(shards)[0]
