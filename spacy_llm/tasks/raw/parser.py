from typing import Iterable, List

from spacy.tokens import Doc

from .task import RawTask


def parse_responses_v1(
    task: RawTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[List[str]]:
    """Parses LLM responses for spacy.Raw.v1. Note that no parsing happens here, as we don't know what the result is
        expected to look like.
    task (RawTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[List[str]]): Reply as string per shard and doc.
    """
    for responses_for_doc in responses:
        yield list(responses_for_doc)
