from typing import Any, Dict, Iterable

from spacy.scorer import Scorer
from spacy.training import Example

from ...compat import BaseModel, Self


class TextCatExample(BaseModel):
    text: str
    answer: str

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        if kwargs["use_binary"]:
            answer = (
                "POS"
                if example.reference.cats[list(kwargs["label_dict"].values())[0]] == 1.0
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

        return TextCatExample(
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
