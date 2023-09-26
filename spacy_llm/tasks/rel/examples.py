from typing import List

from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import RELTask
from .util import EntityItem, RelationItem


class RELExample(FewshotExample[RELTask]):
    text: str
    ents: List[EntityItem]
    relations: List[RelationItem]

    @classmethod
    def generate(cls, example: Example, task: RELTask) -> Self:
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
