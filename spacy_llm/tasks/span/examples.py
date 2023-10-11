import abc
from typing import Dict, Generic, Iterable, List

from spacy.tokens import Span

from ...compat import BaseModel, Self
from ...ty import FewshotExample, TaskContraT


class SpanExample(FewshotExample[TaskContraT], abc.ABC, Generic[TaskContraT]):
    """Example for span tasks not using CoT.
    Note: this should be SpanTaskContraT instead of TaskContraT, but this would entail a circular import.
    """

    text: str
    entities: Dict[str, List[str]]


class SpanReason(BaseModel):
    text: str
    is_entity: bool
    label: str
    reason: str

    @classmethod
    def from_str(cls, line: str, sep: str = "|") -> Self:
        """Parse a single line of LLM output which identifies a span of text,
        whether it's an entity, a label to assign to it, and a reason for that
        assignment.

        e.g. expected line would look like:
        1. Golden State Warriors | True | BASKETBALL_TEAM | is a basketball team in the NBA

        Handles an optional numbered list which we put in the prompt by default so the LLM
        can better structure the order of output for the spans. This number isn't crucial for
        the final parsing so we just strip it for now.

        line (str): Line of LLM output to parse
        sep (str): Optional separator to split on. Defaults to "|".

        RETURNS (Self)
        """
        clean_str = line.strip()
        if "." in clean_str:
            clean_str = clean_str.split(".", maxsplit=1)[1]
        components = [c.strip() for c in clean_str.split(sep)]
        if len(components) != 4:
            raise ValueError(
                "Unable to parse line of LLM output into a SpanReason. ",
                f"line: {line}",
            )

        return cls(
            text=components[0],
            is_entity=components[1].lower() == "true",
            label=components[2],
            reason=components[3],
        )

    def to_str(self, sep: str = "|") -> str:
        """Output as a single line of text representing the expected LLM COT output
        e.g.
        'Golden State Warriors | True | BASKETBALL_TEAM | is a basketball team in the NBA'
        """
        return (
            f"{self.text} {sep} {self.is_entity} {sep} {self.label} {sep} {self.reason}"
        )

    def __str__(self) -> str:
        return self.to_str()


class SpanCoTExample(FewshotExample[TaskContraT], abc.ABC, Generic[TaskContraT]):
    """Example for span tasks using CoT.
    Note: this should be SpanTaskContraT instead of TaskContraT, but this would entail a circular import.
    """

    text: str
    spans: List[SpanReason]

    class Config:
        arbitrary_types_allowed = True

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
