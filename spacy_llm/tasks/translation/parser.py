from typing import Iterable

from spacy.tokens import Doc

from .task import TranslationTask


def parse_responses_v1(
    task: TranslationTask,
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[Iterable[str]]:
    """Parses LLM responses for spacy.Translation.v1.
    task (TranslationTask): Task instance.
    docs (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[str]]): Summary per shard/response.
    """
    for responses_for_doc in responses:
        yield list(responses_for_doc)
