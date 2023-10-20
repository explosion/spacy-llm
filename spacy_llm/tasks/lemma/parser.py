from typing import Iterable, List

from spacy.tokens import Doc

from .task import LemmaTask


def parse_responses_v1(
    task: LemmaTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[Iterable[List[List[str]]]]:
    """Parses LLM responses for spacy.Lemma.v1.
    task (LemmaTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[List[str]]): Lists of 2-lists (token: lemmatized token) per doc/response.
    """
    for responses_for_doc in responses:
        results_for_doc: List[List[List[str]]] = []
        for response in responses_for_doc:
            results_for_doc.append(
                [
                    [pr_part.strip() for pr_part in pr.split(":")]
                    for pr in response.replace("Lemmatized text:", "")
                    .replace("'''", "")
                    .strip()
                    .split("\n")
                ]
            )

        yield results_for_doc
