from ...compat import BaseModel, field_validator


class RelationItem(BaseModel):
    dep: int
    dest: int
    relation: str

    @field_validator("dep", "dest", mode="before")
    @classmethod
    def clean_ent(cls, value):
        if isinstance(value, str):
            value = value.strip("ENT")
        return value


class EntityItem(BaseModel):
    start_char: int
    end_char: int
    label: str
