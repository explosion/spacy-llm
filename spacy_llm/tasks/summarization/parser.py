from typing import Iterable


def parse_responses_v1(responses: Iterable[str]) -> Iterable[str]:
    """Parses LLM responses for spacy.Summarization.v1 and maps them onto Docs.
    responses (Iterable[str]): LLM responses.
    field (str): Field to store responses in.
    RETURNS (Iterable[str]): Summary per doc/response.
    """
    for prompt_response in responses:
        yield prompt_response.replace("'''", "").strip()
