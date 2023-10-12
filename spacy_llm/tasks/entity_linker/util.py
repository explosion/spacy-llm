import csv
import dataclasses
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import spacy
import srsly
from spacy import Vocab
from spacy.kb import InMemoryLookupKB
from spacy.pipeline import EntityLinker
from spacy.scorer import Scorer
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import EntityLinkerTask

UNAVAILABLE_ENTITY_DESC: str = "This entity doesn't have a description."


class ELExample(FewshotExample):
    text: str
    mentions: List[str]
    entity_descriptions: List[List[str]]
    entity_ids: List[List[str]]
    solutions: List[str]
    reasons: Optional[List[str]]

    @property
    def mentions_str(self) -> str:
        """Returns stringified version of all mentions.
        RETURNS (str): Stringified version of all mentions.
        """
        return ", ".join([f"*{mention}*" for mention in self.mentions])

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Optional[Self]:
        # Check whether all entities have their knowledge base IDs set.
        n_ents = len(example.reference.ents)
        n_set_kb_ids = sum([ent.kb_id != 0 for ent in example.reference.ents])
        if n_ents and n_ents != n_set_kb_ids:
            warnings.warn(
                f"Not all entities in this document have their knowledge base IDs set ({n_set_kb_ids} out of "
                f"{n_ents}). Ignoring {n_set_kb_ids - n_ents} entities in example:\n{example.reference}"
            )
        example.reference.ents = [
            ent for ent in example.reference.ents if ent.kb_id != 0
        ]
        if len(example.reference.ents) == 0:
            return None

        # Assemble example.
        mentions = [ent.text for ent in example.reference.ents]
        # Fetch candidates. If true entity not among candidates: fetch description separately and add manually.
        cands_ents, solutions = kwargs["fetch_entity_info"](example.reference)
        # If we are to use available docs as examples, they have to have KB IDs set and hence available solutions.
        assert all([sol is not None for sol in solutions])

        return ELExample(
            text=EntityLinkerTask.highlight_ents_in_text(example.reference),
            mentions=mentions,
            entity_descriptions=[
                [ent.description for ent in ents] for ents in cands_ents
            ],
            entity_ids=[[ent.id for ent in ents] for ents in cands_ents],
            solutions=solutions,
            reasons=[""] * len(mentions),
        )


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score entity linking accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_links(examples, negative_labels=[EntityLinker.NIL])


def ent_desc_reader_csv(path: Union[Path, str]) -> Dict[str, str]:
    """Instantiates entity description reader with two columns: ID and description.
    path (Union[Path, str]): File path.
    RETURNS (Dict[str, str]): Dict with ID -> description.
    """
    with open(path) as csvfile:
        descs: Dict[str, str] = {}
        for row in csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=";"):
            if len(row) != 2:
                continue
            descs[row[0]] = row[1]

        if len(descs) == 0:
            raise ValueError(
                "Format of CSV file with entity descriptions is wrong. CSV has to be formatted as "
                "semicolon-delimited CSV with two columns. The first columns has to contain the entity"
                " ID, the second the entity description."
            )

    return descs


@dataclasses.dataclass
class InMemoryLookupKBLoader:
    """Config/init helper class for InMemoryLookupKB usage in CandidateSelector."""

    path: Union[str, Path]
    nlp_path: Optional[Union[str, Path]]

    def __post_init__(self):
        if self.nlp_path and isinstance(self.nlp_path, str):
            self.nlp_path = Path(self.nlp_path)
        if self.path and isinstance(self.path, str):
            self.path = Path(self.path)

    def __call__(self, vocab: Vocab) -> InMemoryLookupKB:
        """Loads InMemoryLookupKB instance from disk.
        vocab (Vocab): Vocab instance of executing pipeline.
        """
        assert isinstance(self.path, Path) and isinstance(self.nlp_path, Path)

        # If path is directory: assume it's a pah to the serialized KB directory.
        if self.path.is_dir():
            # Load pipeline, use its vocab. If pipeline path isn't set, try loading the surrounding pipeline
            # (which might fail).
            nlp_path = self.nlp_path if self.nlp_path else self.path.parent.parent
            try:
                nlp = spacy.load(nlp_path)
            except IOError as err:
                raise ValueError(
                    f"Pipeline at path {nlp_path} could not be loaded. Make sure to specify the correct path."
                ) from err
            kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
            kb.from_disk(self.path)

        # Otherwise: has to be path to .yaml file.
        else:
            kb_data = srsly.read_yaml(self.path)
            entities = kb_data["entities"]
            qids = list(entities.keys())
            kb = InMemoryLookupKB(
                vocab=vocab,
                entity_vector_length=len(kb_data["entities"][qids[0]]["embedding"]),
            )

            # Set entities (with dummy values for frequencies).
            kb.set_entities(
                entity_list=qids,
                vector_list=[entities[qid]["embedding"] for qid in qids],
                freq_list=[1] * len(qids),
            )

            # Add aliases and dummy prior probabilities.
            for alias_data in kb_data["aliases"]:
                kb.add_alias(**alias_data)

        return kb
