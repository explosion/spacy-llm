from typing import Iterable, List


def parse_responses_v1(responses: Iterable[str]) -> Iterable[List[List[str]]]:
    """Parses LLM responses for spacy.Lemma.v1 and maps them onto Docs.
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
