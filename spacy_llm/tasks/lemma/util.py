from typing import Any, Dict, Iterable, List

from spacy.scorer import Scorer
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample


class LemmaExample(FewshotExample):
    text: str
    lemmas: List[Dict[str, str]]

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        lemma_dict = [{t.text: t.lemma_} for t in example.reference]
        return LemmaExample(text=example.reference.text, lemmas=lemma_dict)


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score lemmatization accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_token_attr(examples, "lemma")
