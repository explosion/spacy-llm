from typing import Iterable

from spacy.tokens import Doc


def parse_responses_v1(docs: Iterable[Doc], responses: Iterable[str]) -> Iterable[Doc]:
    """Parses LLM responses for spacy.lemma.v1 and maps them onto Docs.
    docs (Iterable[Doc]): Docs to map responses to.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[Doc]): Updated docs with mapped LLM responses.
    """
    for doc, prompt_response in zip(docs, responses):
        parsed_response = [
            [pr_part.strip() for pr_part in pr.split(":")]
            for pr in prompt_response.replace("Lemmatized text:", "")
            .replace("'''", "")
            .strip()
            .split("\n")
        ]
        tokens = [token for token in doc]

        # If numbers of tokens recognized by spaCy and returned by LLM don't match, we don't attempt a partial
        # match.
        if len(tokens) != len(parsed_response):
            yield doc

        # Assign lemmas.
        for token, lemma_info in zip(tokens, parsed_response):
            if len(lemma_info) > 0:
                token.lemma_ = lemma_info[1]

        yield doc
