from typing import Callable, Dict, Iterable, List, Optional, Tuple, cast

import jinja2
import srsly
from spacy import util
from spacy.tokens import Doc, Span

from ...compat import Literal
from ...registry import lowercase_normalizer
from .examples import SpanExample
from .parsing import find_substrings


class SpanTask:
    """Base class for Span-related tasks, eg NER and SpanCat."""

    PLAIN_CONFIG_KEYS: List[str] = [
        "_label_dict",
        "_template",
        "_label_definitions",
        "_alignment_mode",
        "_case_sensitive_matching",
        "_single_match",
    ]

    def __init__(
        self,
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]] = {},
        examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {self._normalizer(label): label for label in labels}
        self._template = template
        self._label_definitions = label_definitions
        self._examples = examples
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._single_match = single_match

    def _validate_alignment(self, mode):
        # ideally, this list should be taken from spaCy, but it's not currently exposed from doc.pyx.
        alignment_modes = ("strict", "contract", "expand")
        if mode not in alignment_modes:
            raise ValueError(
                f"Unsupported alignment mode '{mode}'. Supported modes: {', '.join(alignment_modes)}"
            )

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._examples,
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

    def to_bytes(
        self,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """

        def serialize_plain_config():
            plain = {key: getattr(self, key) for key in self.PLAIN_CONFIG_KEYS}
            return srsly.json_dumps(plain)

        def serialize_examples():
            if self._examples is None:
                return srsly.json_dumps(None)
            examples = [eg.dict() for eg in self._examples]
            return srsly.json_dumps(examples)

        serialize = {
            "plain": serialize_plain_config,
            "examples": serialize_examples,
        }

        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "SpanTask":
        """Load the Task from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple[str]): Names of properties to exclude from deserialization.
        RETURNS (SpanTask): Modified SpanTask instance.
        """

        def deserialize_plain_config(b: bytes):
            plain = srsly.json_loads(b)
            for key, value in plain.items():
                setattr(self, key, value)

        def deserialize_examples(b: bytes):
            examples = srsly.json_loads(b)
            if examples is not None:
                self._examples = [SpanExample.parse_obj(eg) for eg in examples]

        deserialize = {
            "plain": deserialize_plain_config,
            "examples": deserialize_examples,
        }

        util.from_bytes(bytes_data, deserialize, exclude)
        return self
