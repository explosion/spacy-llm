from typing import Iterable, Optional, runtime_checkable

from pydantic import BaseModel
from spacy.tokens import Span

from ...compat import Protocol


class EntityCandidate(BaseModel):
    """Represents one entity candidate."""

    id: str
    description: str
    mention: Optional[str]


@runtime_checkable
class CandidateSelector(Protocol):
    def __call__(self, mentions: Iterable[Span]) -> Iterable[Iterable[EntityCandidate]]:
        """Return list of Candidates with their descriptions for given mention and context.
        mentions (Iterable[Span]): Entity mentions.
        RETURNS (Iterable[Iterable[Entity]]): Top n entity candidates per mention.
        """

    def get_entity_description(self, entity_id: str) -> str:
        """Retrieve entity description.
        entity_id (str): Entity ID.
        RETURNS (str): Description for specified entity ID.
        """
