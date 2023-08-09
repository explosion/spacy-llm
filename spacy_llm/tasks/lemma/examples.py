from typing import Dict, List

from spacy.training import Example

from ...ty import FewshotExample


class LemmaExample(FewshotExample):
    text: str
    lemmas: List[Dict[str, str]]

    @classmethod
    def generate(cls, example: Example) -> "LemmaExample":
        lemma_dict = [{t.text: t.lemma_} for t in example.reference]
        return LemmaExample(text=example.reference.text, lemmas=lemma_dict)
