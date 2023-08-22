import re
from typing import Iterable, List

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

    for doc, prompt_response in zip(docs, responses):
        solutions = [
            sol.replace("--- ", "").replace("<", "").replace(">", "")
            for sol in re.findall(r"--- <.*>", prompt_response)
        ]
        # Skip document if the numbers of entities and solutions don't line up.
        if len(solutions) != len(doc.ents):
            yield []

        # Set ents anew by copying them and specifying the KB ID.
        yield [
            Span(
                doc=doc,
                start=ent.start,
                end=ent.end,
                label=ent.label,
                vector=ent.vector,
                vector_norm=ent.vector_norm,
                kb_id=solution,
            )
            for ent, solution in zip(doc.ents, solutions)
        ]
