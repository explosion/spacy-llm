import abc
from typing import Generic, List

from ...compat import BaseModel, validator
from ...ty import FewshotExample, TaskContraT


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


class BaseRELExample(FewshotExample[TaskContraT], abc.ABC, Generic[TaskContraT]):
    text: str
    ents: List[EntityItem]

    class Config:
        arbitrary_types_allowed = True
