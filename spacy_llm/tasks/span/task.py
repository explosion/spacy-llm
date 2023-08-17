import abc
import typing
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Type

from spacy.tokens import Doc, Span

from ...compat import Literal, Self
from ...ty import TaskResponseParser
from ..builtin_task import BuiltinTaskWithLabels
from .util import SpanExample


class SpanTask(BuiltinTaskWithLabels, abc.ABC):
    """Base class for Span-related tasks, eg NER and SpanCat."""

    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[SpanExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[SpanExample]],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],  # noqa: F821
        case_sensitive_matching: bool,
        single_match: bool,
    ):
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            labels=labels,
            label_definitions=label_definitions,
            normalizer=normalizer,
        )

        self._prompt_example_type = typing.cast(
            Type[SpanExample], self._prompt_example_type
        )
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._single_match = single_match

        if self._prompt_examples:
            self._prompt_examples: List[SpanExample] = self._check_label_consistency()

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
                self._prompt_example_type(
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
    def prompt_template(self) -> str:
        return self._template

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        return super().generate_prompts(
            docs=docs,
            labels=list(self._label_dict.values()),
            label_definitions=self._label_definitions,
            **kwargs,
        )

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
        for doc, spans in zip(docs, self._parse_responses(self, docs, responses)):
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
    def alignment_mode(self) -> Literal["strict", "contract", "expand"]:  # noqa: F821
        return self._alignment_mode

    @property
    def case_sensitive_matching(self) -> bool:
        return self._case_sensitive_matching

    @property
    def single_match(self):
        return self._single_match
