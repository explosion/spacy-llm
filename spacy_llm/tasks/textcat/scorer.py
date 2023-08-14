from typing import Any, Dict, Iterable

from spacy.scorer import Scorer
from spacy.training import Example


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score textcat accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_cats(
        examples,
        attr=kwargs["attr"],
        labels=kwargs["labels"],
        multi_label=kwargs["multi_label"],
    )
