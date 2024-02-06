from typing import Iterable, List

from spacy.tokens import Doc
from wasabi import msg

from ...compat import ValidationError
from .task import RELTask
from .util import RelationItem


def parse_responses_v1(
    task: RELTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[Iterable[List[RelationItem]]]:
    """Parses LLM responses for spacy.REL.v1.
    task (RELTask): Task instance.
    docs (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[List[RelationItem]]]): List of RelationItem instances per shard/response.
    """
    for responses_for_doc, shards_for_doc in zip(responses, shards):
        results_for_doc: List[List[RelationItem]] = []
        for response, shard in zip(responses_for_doc, shards_for_doc):
            relations: List[RelationItem] = []
            for line in response.strip().split("\n"):
                try:
                    rel_item = RelationItem.parse_raw(line)
                    if 0 <= rel_item.dep < len(shard.ents) and 0 <= rel_item.dest < len(
                        shard.ents
                    ):
                        relations.append(rel_item)
                except ValidationError:
                    msg.warn(
                        "Validation issue",
                        line,
                        show=task.verbose,
                    )

            results_for_doc.append(relations)

        yield results_for_doc
