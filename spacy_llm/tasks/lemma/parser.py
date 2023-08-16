from typing import Iterable, List

from spacy.tokens import Doc

from .task import LemmaTask


def parse_responses_v1(
    task: LemmaTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[List[List[str]]]:
    """Parses LLM responses for spacy.Lemma.v1.
    task (LemmaTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[List[str]]): Lists of 2-lists (token: lemmatized token) per doc/response.
    """
    for prompt_response in responses:
        yield [
            [pr_part.strip() for pr_part in pr.split(":")]
            for pr in prompt_response.replace("Lemmatized text:", "")
            .replace("'''", "")
            .strip()
            .split("\n")
        ]
