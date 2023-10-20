from typing import Iterable, List, Optional

from spacy.tokens import Doc

from .task import SentimentTask


def parse_responses_v1(
    task: SentimentTask,
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[Iterable[Optional[float]]]:
    """Parses LLM responses for spacy.Sentiment.v1.
    task (SentimentTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[Optional[float]]]): Sentiment score per shard/response. None on parsing error.
    """
    for responses_for_doc in responses:
        results_for_doc: List[Optional[float]] = []
        for response in responses_for_doc:
            try:
                results_for_doc.append(
                    float("".join(response.replace("Answer:", "").strip().split()))
                )
            except ValueError:
                results_for_doc.append(None)

        yield results_for_doc
