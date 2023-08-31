from typing import Iterable, List

from spacy.tokens import Doc
from wasabi import msg

from ...compat import ValidationError
from .task import RELTask
from .util import RelationItem


def parse_responses_v1(
    task: RELTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[List[RelationItem]]:
    """Parses LLM responses for spacy.REL.v1.
    task (RELTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[List[RelationItem]]): List of RelationItem instances per doc/response.
    """
    for response, doc in zip(responses, docs):
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
                    show=task.verbose,
                )

        yield relations
