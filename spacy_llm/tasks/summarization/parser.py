from typing import Iterable

from spacy.tokens import Doc

from .task import SummarizationTask


def parse_responses_v1(
    task: SummarizationTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[str]:
    """Parses LLM responses for spacy.Summarization.v1.
    task (SummarizationTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[str]): Summary per doc/response.
    """
    for prompt_response in responses:
        yield prompt_response.replace("'''", "").strip()
