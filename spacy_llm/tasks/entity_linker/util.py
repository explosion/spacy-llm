import abc
import csv
import dataclasses
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import spacy
import srsly
from spacy import Vocab
from spacy.kb import InMemoryLookupKB
from spacy.pipeline import EntityLinker
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import EntityLinkerTask
from .ty import DescFormat, EntDescReader

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
    def generate(cls, example: Example, task: EntityLinkerTask) -> Optional[Self]:
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
        cands_ents, solutions = task.fetch_entity_info(example.reference)
        # If we are to use available docs as examples, they have to have KB IDs set and hence available solutions.
        assert all([sol is not None for sol in solutions])

        return ELExample(
            text=EntityLinkerTask.highlight_ents_in_doc(example.reference).text,
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
class BaseInMemoryLookupKBLoader:
    path: Union[str, Path]
    """Path to artefact containing knowledge base data."""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    @abc.abstractmethod
    def __call__(self, vocab: Vocab) -> Tuple[InMemoryLookupKB, DescFormat]:
        """Loads KB instance.
        vocab (Vocab): Vocab instance of executing pipeline.
        RETURNS (Tuple[InMemoryLookupKB, DescFormat]): Loaded/generated KB instance; descriptions for entities.
        """
        ...


@dataclasses.dataclass
class KBObjectLoader(BaseInMemoryLookupKBLoader):
    """Config/init helper class for loading InMemoryLookupKB instance from a serialized KB directory."""

    # Path to serialized NLP pipeline. If None, path will be guessed.
    nlp_path: Optional[Union[str, Path]]
    # Path to .csv file with descriptions for entities. Has to have two columns with the first one being the entity ID,
    # the second one being the description. The entity ID has to match with the entity ID in the stored knowledge base.
    # If not specified, all entity descriptions provided in prompts will be a generic "No description available" or
    # something else to this effect.
    desc_path: Optional[Union[Path, str]]
    # Entity description file reader.
    ent_desc_reader: EntDescReader

    def __post_init__(self):
        super().__post_init__()
        if self.nlp_path and isinstance(self.nlp_path, str):
            self.nlp_path = Path(self.nlp_path)

    def __call__(self, vocab: Vocab) -> Tuple[InMemoryLookupKB, DescFormat]:
        assert isinstance(self.path, Path)

        # Load pipeline, use its vocab. If pipeline path isn't set, try loading the surrounding pipeline
        # (which might fail).
        nlp_path = self.nlp_path or self.path.parent.parent
        try:
            nlp = spacy.load(nlp_path)
        except IOError as err:
            raise ValueError(
                f"Pipeline at path {nlp_path} could not be loaded. Make sure to specify the correct path."
            ) from err
        kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
        kb.from_disk(self.path)

        return kb, self.ent_desc_reader(self.desc_path) if self.desc_path else {}


@dataclasses.dataclass
class KBFileLoader(BaseInMemoryLookupKBLoader):
    """Config/init helper class for generating an InMemoryLookupKB instance from a file.
    Currently supports only .yaml files."""

    def __call__(self, vocab: Vocab) -> Tuple[InMemoryLookupKB, DescFormat]:
        assert isinstance(self.path, Path)

        kb_data = srsly.read_yaml(self.path)
        entities = kb_data["entities"]
        qids = list(entities.keys())
        kb = InMemoryLookupKB(
            vocab=vocab,
            entity_vector_length=len(
                kb_data["entities"][qids[0]].get("embedding", [0])
            ),
        )

        # Set entities (with dummy values for frequencies).
        kb.set_entities(
            entity_list=qids,
            # Use [0] as default embedding if no embeddings are specified.
            vector_list=[entities[qid].get("embedding", [0]) for qid in qids],
            freq_list=[1] * len(qids),
        )

        # Add aliases and prior probabilities.
        for alias_data in kb_data["aliases"]:
            try:
                kb.add_alias(**alias_data)
            except ValueError as err:
                if "E134" in str(err):
                    raise ValueError(
                        f"Parsing of YAML file for knowledge base creation failed due to entity in "
                        f"`aliases` section not declared in `entities` section: {alias_data}. Double-"
                        f"check your .yaml file is correct."
                    ) from err
                raise err

        return kb, {qid: entities[qid].get("desc") for qid in qids}


def reduce_shards_to_doc(task: EntityLinkerTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for EntityLinkerTask.
    task (EntityLinkerTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # Entities are additive, so we can just merge shards.
    return Doc.from_docs(list(shards), ensure_whitespace=True)
