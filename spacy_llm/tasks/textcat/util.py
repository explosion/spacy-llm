import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, Optional

from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import TextCatTask


class TextCatExample(FewshotExample[TextCatTask]):
    text: str
    answer: str

    @classmethod
    def generate(cls, example: Example, task: TextCatTask) -> Optional[Self]:
        if task.use_binary:
            answer = (
                "POS"
                if example.reference.cats[list(task.label_dict.values())[0]] == 1.0
                else "NEG"
            )
        else:
            answer = ",".join(
                [
                    label
                    for label, score in example.reference.cats.items()
                    if score == 1.0
                ]
            )

        return cls(
            text=example.reference.text,
            answer=answer,
        )


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score textcat accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_cats(
        examples,
        attr=kwargs["attr"],
        labels=kwargs["labels"],
        multi_label=kwargs["multi_label"],
    )


def reduce_shards_to_doc(task: TextCatTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for TextCatTask.
    task (TextCatTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    shards = list(shards)

    # Compute average sum per category weighted by shard length.
    weights = [len(shard) for shard in shards]
    weights = [n_tokens / sum(weights) for n_tokens in weights]
    all_cats: DefaultDict[str, float] = defaultdict(lambda: 0)
    for weight, shard in zip(weights, shards):
        for cat, cat_score in shard.cats.items():
            all_cats[cat] += cat_score * weight

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping .* while merging docs.",
        )
        doc = Doc.from_docs(shards, ensure_whitespace=True)
    doc.cats = all_cats

    return doc
