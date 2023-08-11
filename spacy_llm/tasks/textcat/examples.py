from pydantic import BaseModel
from spacy.training import Example

from ...compat import Self


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
