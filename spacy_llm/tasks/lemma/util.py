from typing import Dict, Iterable, List

from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample
from ..templates import read_template

DEFAULT_LEMMA_TEMPLATE_V1 = read_template("lemma")


class LemmaExample(FewshotExample):
    text: str
    lemmas: List[Dict[str, str]]

    @classmethod
    def generate(cls, example: Example) -> "LemmaExample":
        lemma_dict = [{t.text: t.lemma_} for t in example.reference]
        return LemmaExample(text=example.reference.text, lemmas=lemma_dict)


def parse_responses_v1(docs: Iterable[Doc], responses: Iterable[str]) -> Iterable[Doc]:
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
