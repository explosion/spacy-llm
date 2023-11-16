import re
from typing import Iterable, List

from spacy.pipeline import EntityLinker
from spacy.tokens import Doc, Span

from .task import EntityLinkerTask


def parse_responses_v1(
    task: EntityLinkerTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[List[Span]]:
    """Parses LLM responses for spacy.EntityLinker.v1.
    task (EntityLinkerTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[List[Span]]): Entity spans per doc.
    """

    for i_doc, (doc, prompt_response) in enumerate(zip(docs, responses)):
        solutions = [
            sol.replace("::: ", "")[1:-1]
            for sol in re.findall(r"::: <.*>", prompt_response)
        ]

        # Set ents anew by copying them and specifying the KB ID.
        ents = [
            ent
            for i_ent, ent in enumerate(doc.ents)
            if task.has_ent_cands[i_doc][i_ent]
        ]
        yield [
            Span(
                doc=doc,
                start=ent.start,
                end=ent.end,
                label=ent.label,
                vector=ent.vector,
                vector_norm=ent.vector_norm,
                kb_id=solution if solution != "NIL" else EntityLinker.NIL,
            )
            for ent, solution in zip(ents, solutions)
        ]
