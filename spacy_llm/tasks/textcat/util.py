from typing import Any, Dict, Iterable, Optional

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


def reduce_shards_to_doc(shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for TextCatTask.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # todo this is yet a dummy implementation that will only return the first doc shard.
    return list(shards)[0]
