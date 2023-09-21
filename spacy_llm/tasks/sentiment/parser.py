from typing import Iterable, Optional

from spacy.tokens import Doc

from .task import SentimentTask


def parse_responses_v1(
    task: SentimentTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[Optional[float]]:
    """Parses LLM responses for spacy.Sentiment.v1.
    task (SentimentTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[Optional[float]]): Sentiment score per doc/response. None on parsing error.
    """
    for prompt_response in responses:
        try:
            yield float("".join(prompt_response.replace("Answer:", "").strip().split()))
        except ValueError:
            yield None
