import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import jinja2
from pydantic import BaseModel
from spacy.tokens import Doc, Span

from ..compat import Literal, Self
from ..registry import lowercase_normalizer
from .util.parsing import find_substrings
from .util.serialization import SerializableTask


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


class SpanExample(BaseModel):
    text: str
    spans: List[SpanReason]


class SpanTask(SerializableTask[SpanExample]):
    """Base class for Span-related tasks, eg NER and SpanCat."""

    def __init__(
        self,
        labels: List[str],
        template: str,
        prompt_examples: List[SpanExample],
        description: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        allow_overlap: Optional[bool] = True,
    ):
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }
        self._template = template
        self._prompt_examples = prompt_examples
        self._description = description
        self._label_definitions = label_definitions
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._allow_overlap = allow_overlap

        if self._prompt_examples:
            self._prompt_examples = self._check_label_consistency()

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self._label_dict.values())

    @property
    def prompt_template(self) -> str:
        return self._template

    def _check_label_consistency(self) -> List[SpanExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.
        RETURNS (List[SpanExample]): List of SpanExamples with valid labels.
        """
        assert self._prompt_examples
        null_labels = {
            self._normalizer(entity.label): entity.label
            for example in self._prompt_examples
            for entity in example.spans
            if not entity.is_entity
        }
        if len(null_labels) > 1:
            warnings.warn(
                f"Negative examples contain multiple negative labels: {', '.join(null_labels.keys())}."
            )

        example_labels = {
            self._normalizer(entity.label): entity.label
            for example in self._prompt_examples
            for entity in example.spans
            if entity.is_entity
        }

        unspecified_labels = {
            example_labels[key]
            for key in (set(example_labels.keys()) - set(self._label_dict.keys()))
        }
        if not set(example_labels.keys()) <= set(self._label_dict.keys()):
            warnings.warn(
                f"Examples contain labels that are not specified in the task configuration. The latter contains the "
                f"following labels: {sorted(list(set(self._label_dict.values())))}. Labels in examples missing from "
                f"the task configuration: {sorted(list(unspecified_labels))}. Please ensure your label specification "
                f"and example labels are consistent."
            )

        # Return examples without non-declared labels. If an example only has undeclared labels, it is discarded.
        include_labels = dict(self._label_dict)
        include_labels.update(null_labels)
        return [
            example
            for example in [
                SpanExample(
                    text=example.text,
                    spans=[
                        entity
                        for entity in example.spans
                        if self._normalizer(entity.label) in include_labels
                    ],
                )
                for example in self._prompt_examples
            ]
            if len(example.spans)
        ]

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                description=self._description,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._prompt_examples,
                allow_overlap=self._allow_overlap,
            )
            yield prompt

    @staticmethod
    def _validate_alignment(alignment_mode: str):
        """Raises error if specified alignment_mode is not supported.
        alignment_mode (str): Alignment mode to check.
        """
        # ideally, this list should be taken from spaCy, but it's not currently exposed from doc.pyx.
        alignment_modes = ("strict", "contract", "expand")
        if alignment_mode not in alignment_modes:
            raise ValueError(
                f"Unsupported alignment mode '{alignment_mode}'. Supported modes: {', '.join(alignment_modes)}"
            )

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        raise NotImplementedError()

    def _extract_span_reasons(self, response: str) -> List[SpanReason]:
        """Parse raw string response into a list of SpanReasons
        response (str): Raw string response from the LLM.
        RETURNS (List[SpanReason]): List of SpanReasons parsed from the response.
        """
        span_reasons = []
        for line in response.strip().split("\n"):
            try:
                span_reason = SpanReason.from_str(line)
            except ValueError:
                continue
            if not span_reason.is_entity:
                continue
            norm_label = self._normalizer(span_reason.label)
            if norm_label not in self._label_dict:
                continue
            label = self._label_dict[norm_label]
            span_reason.label = label
            span_reasons.append(span_reason)
        return span_reasons

    def _find_spans(self, doc: Doc, span_reasons: List[SpanReason]) -> List[Span]:
        """Find a list of spaCy Spans from a list of SpanReasons
        for a single spaCy Doc

        doc (Doc): Input doc to parse spans for
        span_reasons (List[SpanReason]): List of SpanReasons to find in doc
        RETURNS (List[Span]): List of spaCy Spans found in doc
        """
        find_after = 0
        spans = []
        prev_span = None
        n_span_reasons = len(span_reasons)
        idx = 0
        while idx < n_span_reasons:
            span_reason = span_reasons[idx]

            # For each phrase, find the SpanReason substring in the text
            # and create a Span
            offsets = find_substrings(
                doc.text,
                [span_reason.text],
                case_sensitive=self._case_sensitive_matching,
                single_match=True,
                find_after=find_after,
            )
            if not offsets:
                idx += 1
                continue

            # Must have exactly one match because single_match=True
            assert len(offsets) == 1
            start, end = offsets[0]

            span = doc.char_span(
                start,
                end,
                alignment_mode=self._alignment_mode,
                label=span_reason.label,
            )

            if span is None:
                # If we couldn't resolve a span, just skip to the next
                # span_reason
                idx += 1
                continue

            if span == prev_span:
                # If the span is the same as the previous span,
                # re-run this span_reason but look farther into the text
                find_after = span.end_char
                continue

            spans.append(span)
            find_after = span.start_char if self._allow_overlap else span.end_char
            prev_span = span
            idx += 1
        return sorted(set(spans))

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        """Since we provide entities in a numbered list, we expect the LLM to
        output entities in the order they occur in the text. This parse
        function now incrementally finds substrings in the text and tracks the
        last found span's start character to ensure we don't overwrite
        previously found spans.
        """
        for doc, llm_response in zip(docs, responses):
            span_reasons = self._extract_span_reasons(llm_response)
            spans = self._find_spans(doc, span_reasons)
            self.assign_spans(doc, spans)
            yield doc

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_spans_key",
            "_label_dict",
            "_template",
            "_label_definitions",
            "_alignment_mode",
            "_case_sensitive_matching",
        ]

    @property
    def _Example(self) -> Type[SpanExample]:
        return SpanExample
