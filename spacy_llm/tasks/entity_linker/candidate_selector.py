import warnings
from typing import Dict, Iterable, Optional

from spacy import Vocab
from spacy.kb import InMemoryLookupKB
from spacy.pipeline import EntityLinker
from spacy.tokens import Span

from .ty import Entity, InMemoryLookupKBLoader
from .util import UNAVAILABLE_ENTITY_DESC


class KBCandidateSelector:
    """Initializes a spaCy InMemoryLookupKB and uses its candidate selection mechanism to return entity candidates."""

    def __init__(
        self,
        kb_loader: InMemoryLookupKBLoader,
        top_n: int,
    ):
        """Generates KBCandidateSelector. Note that this class has to be initialized (.initialize()) before being used.
        kb_loader (InMemoryLookupKBLoader): KB loader.
        top_n (int): Top n candidates to include in prompt.
        """
        self._kb_loader = kb_loader
        self._kb: Optional[InMemoryLookupKB] = None
        self._descs: Dict[str, str] = {}
        self._top_n = top_n

    def initialize(self, vocab: Vocab) -> None:
        """Initialize instance with vocabulary.
        vocab (Vocab): Vocabulary.
        """
        self._kb, self._descs = self._kb_loader(vocab)

    def __call__(self, mentions: Iterable[Span]) -> Iterable[Iterable[Entity]]:
        """Retrieves top n candidates using spaCy's entity linker's .get_candidates_batch().
        mentions (Iterable[Span]): Mentions to look up entity candidates for.
        RETURNS (Iterable[Iterable[Entity]]): Top n entity candidates per mention.
        """
        if self._kb is None:
            raise ValueError("CandidateSelector has to be initialized before usage.")

        all_cands = self._kb.get_candidates_batch(mentions)
        for cands in all_cands:
            assert isinstance(cands, list)
            cands.sort(key=lambda x: x.prior_prob, reverse=True)

        return [
            [
                Entity(
                    id=cand.entity_,
                    description=self.get_entity_description(cand.entity_),
                )
                for cand in cands[: self._top_n]
            ]
            if len(cands) > 0
            else [Entity(id=EntityLinker.NIL, description=UNAVAILABLE_ENTITY_DESC)]
            for cands in all_cands
        ]

    def get_entity_description(self, entity_id: str) -> str:
        """Returns entity description for entity ID. If none found, a warning is emitted and
            spacy_llm.tasks.entity_linker.util.UNAVAILABLE_ENTITY_DESC is returned.
        entity_id (str): Entity whose ID should be looked up.
        RETURNS (str): Entity description for entity with specfied ID. If no description found, returned string equals
            spacy_llm.tasks.entity_linker.util.UNAVAILABLE_ENTITY_DESC.
        """
        if entity_id not in self._descs:
            warnings.warn(
                f"Entity with ID {entity_id} is not in provided descriptions."
            )

        return self._descs.get(entity_id, UNAVAILABLE_ENTITY_DESC)
