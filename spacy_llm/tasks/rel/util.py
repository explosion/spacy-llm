import warnings
from typing import Iterable, List, Optional

from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .items import EntityItem, RelationItem
from .task import RELTask


class RELExample(FewshotExample[RELTask]):
    text: str
    ents: List[EntityItem]
    relations: List[RelationItem]

    @classmethod
    def generate(cls, example: Example, task: RELTask) -> Optional[Self]:
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


def reduce_shards_to_doc(task: RELTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for RELTask.
    task (RELTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    shards = list(shards)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=f".*Skipping Doc custom extension '{task.field}' while merging docs.",
        )
        doc = Doc.from_docs(shards, ensure_whitespace=True)

    # REL information from shards can be simply appended.
    setattr(
        doc._,
        task.field,
        [rel_items for shard in shards for rel_items in getattr(shard._, task.field)],
    )

    return doc
