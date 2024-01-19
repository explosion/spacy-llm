from typing import Iterable, List

from spacy.tokens import Doc

from .task import SummarizationTask


def parse_responses_v1(
    task: SummarizationTask,
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[Iterable[str]]:
    """Parses LLM responses for spacy.Summarization.v1.
    task (SummarizationTask): Task instance.
    docs (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[str]]): Summary per shard/response.
    """
    for responses_for_doc in responses:
        results_for_doc: List[str] = []
        for response in responses_for_doc:
            results_for_doc.append(response.replace("'''", "").strip())

        yield responses_for_doc
