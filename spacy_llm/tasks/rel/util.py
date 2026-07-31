import re
import warnings
from typing import Iterable, List, Optional

from spacy import Vocab
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

    def to_doc(self) -> Doc:
        """Returns Doc representation of example instance. Note that relations are in user_data["rel"].
        field (str): Doc field to store relations in.
        RETURNS (Doc): Representation as doc.
        """
        punct_chars_pattern = r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+'
        text = re.sub(punct_chars_pattern, r" \g<0> ", self.text)
        doc_words = text.split()
        doc_spaces = [
            i < len(doc_words) - 1
            and not re.match(punct_chars_pattern, doc_words[i + 1])
            for i, word in enumerate(doc_words)
        ]
        doc = Doc(words=doc_words, spaces=doc_spaces, vocab=Vocab(strings=doc_words))
        char_offsets = _map_char_offsets(self.text, doc.text)

        # Set entities using offsets from the original, unnormalized text.
        doc.ents = [
            doc.char_span(
                char_offsets[entity.start_char],
                char_offsets[entity.end_char],
                label=entity.label,
            )
            for entity in self.ents
        ]
        doc.user_data["rel"] = self.relations

        return doc


def _map_char_offsets(source: str, target: str) -> List[int]:
    """Map source character boundaries to a target with inserted spaces."""
    offsets = [0] * (len(source) + 1)
    source_index = 0
    target_index = 0

    while source_index < len(source):
        while (
            target_index < len(target)
            and target[target_index] == " "
            and target[target_index] != source[source_index]
        ):
            target_index += 1
        if target_index >= len(target) or target[target_index] != source[source_index]:
            raise ValueError("Unable to map normalized text offsets")
        offsets[source_index] = target_index
        source_index += 1
        target_index += 1

    offsets[len(source)] = target_index
    return offsets


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
            message=".*Skipping .* while merging docs.",
        )
        doc = Doc.from_docs(shards, ensure_whitespace=True)

    # REL information from shards can be simply appended.
    setattr(
        doc._,
        task.field,
        [rel_items for shard in shards for rel_items in getattr(shard._, task.field)],
    )

    return doc
