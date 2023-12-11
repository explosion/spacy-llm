from typing import Iterable, List

from spacy.tokens import Doc

from .task import LemmaTask


def parse_responses_v1(
    task: LemmaTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[List[List[List[str]]]]:
    """Parses LLM responses for spacy.Lemma.v1.
    task (LemmaTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[List[List[List[str]]]]): Lists of 2-lists per token (token: lemmatized token) and shard/response
        and doc.
    """
    for responses_for_doc in responses:
        results_for_doc: List[List[List[str]]] = []
        for response in responses_for_doc:
            results_for_shard = [
                [pr_part.strip() for pr_part in pr.split(":")]
                for pr in response.replace("Lemmatized text:", "")
                .replace("'''", "")
                .strip()
                .split("\n")
            ]
            results_for_doc.append(
                # Malformed responses might have a length != 2, in which case they are discarded.
                [
                    result_for_token
                    for result_for_token in results_for_shard
                    if len(result_for_token) == 2
                ]
            )

        yield results_for_doc
