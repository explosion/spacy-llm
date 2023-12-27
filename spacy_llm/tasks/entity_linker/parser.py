import re
from typing import Iterable, List

from spacy.pipeline import EntityLinker
from spacy.tokens import Doc, Span

from .task import EntityLinkerTask


def parse_responses_v1(
    task: EntityLinkerTask,
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[List[List[Span]]]:
    """Parses LLM responses for spacy.EntityLinker.v1.
    task (EntityLinkerTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[List[List[Span]]): Entity spans per shard.
    """

    for i_doc, (shards_for_doc, responses_for_doc) in enumerate(zip(shards, responses)):
        results_for_doc: List[List[Span]] = []
        for i_shard, (shard, response) in enumerate(
            zip(shards_for_doc, responses_for_doc)
        ):
            solutions = [
                sol.replace("::: ", "")[1:-1]
                for sol in re.findall(r"::: <.*>", response)
            ]

            # Set ents anew by copying them and specifying the KB ID.
            ents = [
                ent
                for i_ent, ent in enumerate(shard.ents)
                if task.has_ent_cands_by_shard[i_doc][i_shard][i_ent]
            ]

            results_for_doc.append(
                [
                    Span(
                        doc=shard,
                        start=ent.start,
                        end=ent.end,
                        label=ent.label,
                        vector=ent.vector,
                        vector_norm=ent.vector_norm,
                        kb_id=solution if solution != "NIL" else EntityLinker.NIL,
                    )
                    for ent, solution in zip(ents, solutions)
                ]
            )

        yield results_for_doc
