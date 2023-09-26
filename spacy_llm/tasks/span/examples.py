import abc
from typing import Dict, Generic, Iterable, List

from spacy.tokens import Span

from ...ty import FewshotExample
from ..ner.util import SpanReason
from .task import SpanTaskContraT


class SpanExample(FewshotExample[SpanTaskContraT], abc.ABC, Generic[SpanTaskContraT]):
    text: str
    entities: Dict[str, List[str]]


class SpanCoTExample(
    FewshotExample[SpanTaskContraT], abc.ABC, Generic[SpanTaskContraT]
):
    text: str
    spans: List[SpanReason]

    @staticmethod
    def _extract_span_reasons(spans: Iterable[Span]) -> List[SpanReason]:
        """Extracts SpanReasons from spans.
        spans (Iterable[Span]): Spans to extract reasons from.
        RETURNS (List[SpanReason]): SpanReason instances extracted from example.
        """
        span_reasons: List[SpanReason] = []
        for span in spans:
            span_reasons.append(
                SpanReason(
                    text=span.text,
                    is_entity=True,
                    label=span.label_,
                    reason=f"is a {span.label_}",
                )
            )

        return span_reasons
