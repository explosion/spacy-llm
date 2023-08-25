from typing import Iterable

from pydantic import BaseModel
from spacy.tokens import Span

from ...compat import Protocol, runtime_checkable


class Entity(BaseModel):
    """Represents one entity."""

    id: str
    description: str


@runtime_checkable
class CandidateSelector(Protocol):
    def __call__(self, mentions: Iterable[Span]) -> Iterable[Iterable[Entity]]:
        """Return list of Candidates with their descriptions for given mention and context.
        mentions (Iterable[Span]): Entity mentions.
        RETURNS (Iterable[Iterable[Entity]]): Top n entity candidates per mention.
        """

    def get_entity_description(self, entity_id: str) -> str:
        """Returns entity description for entity ID. If none found, a warning is emitted and
        spacy_llm.tasks.enttiy_linker.util.UNAVAILABLE_ENTITY_DES is returned.
        entity_id (str): Entity whose ID should be looked up.
        RETURNS (str): Entity description for entity with specfied ID. If no description found, returned string equals
            spacy_llm.tasks.enttiy_linker.util.UNAVAILABLE_ENTITY_DESC.
        """
