from typing import Iterable

from spacy.tokens import Doc

from ...compat import BaseModel, validator


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


def reduce_shards_to_doc(shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for RELTask.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # todo this is yet a dummy implementation that will only return the first doc shard.
    return list(shards)[0]
