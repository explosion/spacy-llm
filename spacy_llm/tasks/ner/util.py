from collections import defaultdict
from typing import Any, Dict, Iterable

from pydantic import BaseModel
from spacy.scorer import get_ner_prf
from spacy.training import Example

from ...compat import Self
from ..span import SpanExample
from ..span.examples import SpanCoTExample
from .task import NERTask


class NERExample(SpanExample[NERTask]):
    @classmethod
    def generate(cls, example: Example, task: NERTask) -> Self:
        entities = defaultdict(list)
        for ent in example.reference.ents:
            entities[ent.label_].append(ent.text)

        return cls(text=example.reference.text, entities=entities)


class NERCoTExample(SpanCoTExample[NERTask]):
    @classmethod
    def generate(cls, example: Example, task: NERTask) -> Self:
        return cls(
            text=example.reference.text,
            spans=SpanCoTExample._extract_span_reasons(example.reference.ents),
        )


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


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score NER accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
    return get_ner_prf(examples)
