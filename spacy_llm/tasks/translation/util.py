import warnings
from typing import Iterable, Optional

from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample
from .task import TranslationTask


class TranslationExample(FewshotExample[TranslationTask]):
    text: str
    translation: str

    @classmethod
    def generate(cls, example: Example, task: TranslationTask) -> Optional[Self]:
        return cls(
            text=example.reference.text,
            summary=getattr(example.reference._, task.field),
        )


def reduce_shards_to_doc(task: TranslationTask, shards: Iterable[Doc]) -> Doc:
    """Reduces shards to docs for TranslationTask.
    task (TranslationTask): Task.
    shards (Iterable[Doc]): Shards to reduce to single doc instance.
    RETURNS (Doc): Fused doc instance.
    """
    shards = list(shards)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping .* while merging docs.",
        )
        doc = Doc.from_docs(list(shards), ensure_whitespace=True)

    setattr(
        doc._, task.field, " ".join([getattr(shard._, task.field) for shard in shards])
    )

    return doc
