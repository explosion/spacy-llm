import re
import warnings
from typing import Iterable, List, Optional, Tuple

from spacy import Vocab
from spacy.tokens import Doc, Span
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

        # Set entities after finding correct indices.
        conv_ent_indices: List[Tuple[int, int]] = []
        if len(self.ents):
            ent_idx = 0
            for token in doc:
                if token.idx == self.ents[ent_idx].start_char:
                    conv_ent_indices.append((token.i, -1))
                if token.idx + len(token.text) == self.ents[ent_idx].end_char:
                    conv_ent_indices[-1] = (conv_ent_indices[-1][0], token.i + 1)
                    ent_idx += 1
                if ent_idx == len(self.ents):
                    break

        doc.ents = [
            Span(  # noqa: E731
                doc=doc,
                start=ent_idx[0],
                end=ent_idx[1],
                label=self.ents[i].label,
            )
            for i, ent_idx in enumerate(conv_ent_indices)
        ]
        doc.user_data["rel"] = self.relations

        return doc


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
