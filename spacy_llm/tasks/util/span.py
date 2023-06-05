from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, cast

import jinja2
import srsly
from spacy import util
from spacy.tokens import Doc, Span

from ...compat import Literal
from ...registry import lowercase_normalizer
from .examples import SpanExample
from .parsing import find_substrings


def serialize_cfg(task: "SpanTask", cfg_keys: List[str]) -> str:
    """Serialize task config.

    task (SpanTask): Task to serialize config from.
    cfg_keys (List[str]): Keys to serialize.
    """
    cfg = {key: getattr(task, key) for key in cfg_keys}
    return srsly.json_dumps(cfg)


def deserialize_cfg(b: str, task: "SpanTask") -> None:
    """Deserialize task config from bytes.

    b (str): serialized config.
    task (SpanTask): Task to set the config on.
    """
    cfg = srsly.json_loads(b)
    for key, value in cfg.items():
        setattr(task, key, value)


def serialize_examples(task: "SpanTask") -> str:
    """Serialize examples.

    task (SpanTask): Task to serialize examples from.
    """
    if task._examples is None:
        return srsly.json_dumps(None)
    examples = [eg.dict() for eg in task._examples]
    return srsly.json_dumps(examples)


def deserialize_examples(b: str, task: "SpanTask"):
    """Deserialize examples from bytes.

    b (str): serialized examples.
    task (SpanTask): Task to set the examples on.
    """
    examples = srsly.json_loads(b)
    if examples is not None:
        task._examples = [SpanExample.parse_obj(eg) for eg in examples]


class SpanTask:
    """Base class for Span-related tasks, eg NER and SpanCat."""

    _CFG_KEYS: List[str] = [
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

        serialize = {
            "cfg": partial(serialize_cfg, task=self, cfg_keys=self._CFG_KEYS),
            "examples": partial(serialize_examples, task=self),
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

        deserialize = {
            "cfg": partial(deserialize_cfg, task=self),
            "examples": partial(deserialize_examples, task=self),
        }

        util.from_bytes(bytes_data, deserialize, exclude)
        return self

    def to_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> None:
        """Serialize the task to disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """

        serialize = {
            "cfg.json": lambda p: p.write_text(
                serialize_cfg(task=self, cfg_keys=self._CFG_KEYS)
            ),
            "examples.json": lambda p: p.write_text(serialize_examples(task=self)),
        }

        util.to_disk(path, serialize, exclude)

    def from_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "SpanTask":
        """Deserialize the task from disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """

        deserialize = {
            "cfg.json": lambda p: deserialize_cfg(p.read_text(), task=self),
            "examples.json": lambda p: deserialize_examples(p.read_text(), task=self),
        }

        util.from_disk(path, deserialize, exclude)
        return self
