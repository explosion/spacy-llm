import warnings
from typing import Any, Dict, Iterable, List, Optional

from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import LemmaTask


class LemmaExample(FewshotExample[LemmaTask]):
    text: str
    lemmas: List[Dict[str, str]]

    @classmethod
    def generate(cls, example: Example, task: LemmaTask) -> Optional[Self]:
        lemma_dict = [{t.text: t.lemma_} for t in example.reference]
        return cls(text=example.reference.text, lemmas=lemma_dict)


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score lemmatization accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return Scorer.score_token_attr(examples, "lemma")


def reduce_shards_to_doc(task: LemmaTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for LemmaTask.
    task (LemmaTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    # Lemmas are token-specific, so we can just merge shards.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping .* while merging docs.",
        )
        return Doc.from_docs(list(shards), ensure_whitespace=True)
