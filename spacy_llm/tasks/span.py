import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import jinja2
from pydantic import BaseModel
from spacy.tokens import Doc, Span

from ..compat import Literal
from ..registry import lowercase_normalizer
from .util.parsing import find_substrings
from .util.serialization import SerializableTask


class SpanExample(BaseModel):
    text: str
    entities: Dict[str, List[str]]


class SpanTask(SerializableTask[SpanExample]):
    """Base class for Span-related tasks, eg NER and SpanCat."""

    def __init__(
        self,
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]] = {},
        prompt_examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }
        self._template = template
        self._label_definitions = label_definitions
        self._prompt_examples = prompt_examples or []
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._single_match = single_match

        if self._prompt_examples:
            self._prompt_examples = self._check_label_consistency()

    def _check_label_consistency(self) -> List[SpanExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.
        RETURNS (List[SpanExample]): List of SpanExamples with valid labels.
        """
        assert self._prompt_examples
        example_labels = {
            self._normalizer(key): key
            for example in self._prompt_examples
            for key in example.entities
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
        return [
            example
            for example in [
                SpanExample(
                    text=example.text,
                    entities={
                        label: entities
                        for label, entities in example.entities.items()
                        if self._normalizer(label) in self._label_dict
                    },
                )
                for example in self._prompt_examples
            ]
            if len(example.entities)
        ]

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self._label_dict.values())

    @property
    def prompt_template(self) -> str:
        return self._template

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._prompt_examples,
            )
            yield prompt

    def _format_response(self, response: str) -> Iterable[Tuple[str, Iterable[str]]]:
        """Parse raw string response into a structured format"""
        output = []
        assert self._normalizer is not None
        for line in response.strip().split("\n"):
            # Check if the formatting we want exists
            # <entity label>: ent1, ent2
            if line and ":" in line:
                label, phrases = line.split(":", 1)
                norm_label = self._normalizer(label)
                if norm_label in self._label_dict:
                    # Get the phrases / spans for each entity
                    if phrases.strip():
                        _phrases = [p.strip() for p in phrases.strip().split(",")]
                        output.append((self._label_dict[norm_label], _phrases))
        return output

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

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            spans = []
            for label, phrases in self._format_response(prompt_response):
                # For each phrase, find the substrings in the text
                # and create a Span
                offsets = find_substrings(
                    doc.text,
                    phrases,
                    case_sensitive=self._case_sensitive_matching,
                    single_match=self._single_match,
                )
                for start, end in offsets:
                    span = doc.char_span(
                        start, end, alignment_mode=self._alignment_mode, label=label
                    )
                    if span is not None:
                        spans.append(span)

            self.assign_spans(doc, spans)
            yield doc

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_alignment_mode",
            "_case_sensitive_matching",
            "_single_match",
        ]

    @property
    def _Example(self) -> Type[SpanExample]:
        return SpanExample
