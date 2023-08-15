from typing import Any, Dict, Iterable

from spacy.scorer import Scorer
from spacy.training import Example


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score lemmatization accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_token_attr(examples, "lemma")
