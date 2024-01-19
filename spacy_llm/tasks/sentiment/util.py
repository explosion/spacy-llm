import warnings
from typing import Any, Dict, Iterable, Optional

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


def reduce_shards_to_doc(task: SentimentTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for SentimentTask by computing an average sentiment score weighted by shard lengths.
    task (SentimentTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    shards = list(shards)
    weights = [len(shard) for shard in shards]
    weights = [n_tokens / sum(weights) for n_tokens in weights]
    sent_scores = [getattr(shard._, task.field) for shard in shards]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping .* while merging docs.",
        )
        doc = Doc.from_docs(shards, ensure_whitespace=True)
    setattr(
        doc._,
        task.field,
        sum([score * weight for score, weight in zip(sent_scores, weights)]),
    )

    return doc


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
