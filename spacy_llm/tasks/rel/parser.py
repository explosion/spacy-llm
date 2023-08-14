from typing import Iterable, List

from wasabi import msg

from spacy_llm.tasks.rel.examples import RelationItem

try:
    from pydantic.v1 import ValidationError
except ImportError:
    from pydantic import ValidationError


def parse_responses_v1(
    responses: Iterable[str], **kwargs
) -> Iterable[List[RelationItem]]:
    """Parses LLM responses for spacy.REL.v1.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[List[RelationItem]]): List of RelationItem instances per doc/response.
    """
    for response, doc in zip(responses, kwargs["docs"]):
        relations: List[RelationItem] = []
        for line in response.strip().split("\n"):
            try:
                rel_item = RelationItem.parse_raw(line)
                if 0 <= rel_item.dep < len(doc.ents) and 0 <= rel_item.dest < len(
                    doc.ents
                ):
                    relations.append(rel_item)
            except ValidationError:
                msg.warn(
                    "Validation issue",
                    line,
                    show=kwargs["verbose"],
                )

        yield relations
