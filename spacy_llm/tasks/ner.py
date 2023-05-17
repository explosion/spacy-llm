from typing import Any, Callable, Iterable, List, Optional, Tuple

import jinja2
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from ..compat import Literal
from ..registry import lowercase_normalizer, registry
from .util import SpanExample, find_substrings


@registry.llm_tasks("spacy.NER.v1")
class NERTask:
    _TEMPLATE_STR = """
From the text below, extract the following entities in the following format:
{# whitespace #}
{%- for label in labels -%}
{{ label }}: <comma delimited list of strings>
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{%- if examples -%}
{# whitespace #}
Below are some examples (only use these as a guide):
{# whitespace #}
{# whitespace #}
{%- for example in examples -%}
{# whitespace #}
Text:
'''
{{ example.text }}
'''
{# whitespace #}
{%- for label, substrings in example.entities.items() -%}
{{ label }}: {{ ', '.join(substrings) }}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{# whitespace #}
{%- endfor -%}
{%- endif -%}
{# whitespace #}
Here is the text that needs labeling:
{# whitespace #}
Text:
'''
{{ text }}
'''
    """

    def __init__(
        self,
        labels: str,
        examples: Optional[Callable[[], Iterable[Any]]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default NER task.

        labels (str): Comma-separated list of labels to pass to the template.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        """
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in labels.split(",")
        }
        self._examples = [SpanExample(**eg) for eg in examples()] if examples else None
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
        _template = environment.from_string(self._TEMPLATE_STR)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
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
    ) -> Doc:
        """Assign spans to the document."""
        doc.set_ents(filter_spans(spans))
        return doc

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
