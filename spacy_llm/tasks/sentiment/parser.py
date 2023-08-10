from typing import Iterable, Optional


def parse_responses_v1(responses: Iterable[str]) -> Iterable[Optional[float]]:
    """Parses LLM responses for spacy.Sentiment.v1.
    responses (Iterable[str]): LLM responses.
    field (str): Field to store responses in.
    RETURNS (Iterable[Optional[float]]): Sentiment score per doc/response. None on parsing error.
    """
    for prompt_response in responses:
        try:
            yield float("".join(prompt_response.replace("Answer:", "").strip().split()))
        except ValueError:
            yield None
