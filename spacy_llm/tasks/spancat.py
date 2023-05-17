from typing import Any, Callable, Iterable, List, Optional

from spacy.tokens import Doc, Span

from ..compat import Literal
from ..registry import registry
from .ner import NERTask
from .templates import read_template


@registry.llm_tasks("spacy.SpanCat.v1")
class SpanCatTask(NERTask):
    _TEMPLATE_STR = read_template("spancat")

    def __init__(
        self,
        labels: str,
        spans_key: str = "sc",
        examples: Optional[Callable[[], Iterable[Any]]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default SpanCat task.

        labels (str): Comma-separated list of labels to pass to the template.
        spans_key (str): Key of the `Doc.spans` dict to save under.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        """
        super(SpanCatTask, self).__init__(
            labels=labels,
            examples=examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )
        self._spans_key = spans_key

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> Doc:
        """Assign spans to the document."""
        doc.spans[self._spans_key] = spans
        return doc
