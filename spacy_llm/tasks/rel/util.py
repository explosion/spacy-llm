from typing import List

from spacy.training import Example

from ...compat import BaseModel, Self, validator
from ...ty import FewshotExample


class RelationItem(BaseModel):
    dep: int
    dest: int
    relation: str

    @validator("dep", "dest", pre=True)
    def clean_ent(cls, value):
        if isinstance(value, str):
            value = value.strip("ENT")
        return value


class EntityItem(BaseModel):
    start_char: int
    end_char: int
    label: str


class RELExample(FewshotExample):
    text: str
    ents: List[EntityItem]
    relations: List[RelationItem]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        entities = [
            EntityItem(
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=ent.label_,
            )
            for ent in example.reference.ents
        ]
        return cls(
            text=example.reference.text,
            ents=entities,
            relations=example.reference._.rel,
        )
