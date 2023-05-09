from typing import Callable, Iterable, Optional, Tuple, Literal

import jinja2
from spacy.tokens import Doc
from spacy.util import filter_spans

from ..registry import noop_normalizer, registry


def find_substrings(
    text: str,
    substrings: Iterable[str],
    *,
    case_sensitive: bool = False,
    single_match: bool = False,
) -> Iterable[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions
    in a text"""

    def _unique(items: Iterable[str]) -> Iterable[str]:
        """Remove duplicates without changing order"""
        seen = set()
        output = []
        for item in items:
            if item not in seen:
                output.append(item)
                seen.add(item)
        return output

    # Remove empty and duplicate strings, and lowercase everything if need be
    substrings = [s for s in substrings if s and len(s) > 0]
    if not case_sensitive:
        text = text.lower()
        substrings = [s.lower() for s in substrings]
    substrings = _unique(substrings)
    offsets = []
    for substring in substrings:
        search_from = 0
        # Search until one hit is found. Continue only if single_match is False.
        while True:
            start = text.find(substring, search_from)
            if start == -1:
                break
            end = start + len(substring)
            offsets.append((start, end))
            if single_match:
                break
            search_from = end
    return offsets


@registry.llm_tasks("spacy.NERZeroShot.v1")
class NerTask:

    _TEMPLATE_STR = """
    From the text below, extract the following entities in the following format:
    {# whitespace #}
    {%- for label in labels -%}
    {{ label }}: <comma delimited list of strings>
    {# whitespace #}
    {%- endfor -%}
    Text:
    '''
    {{ text }}
    """

    def __init__(
        self,
        labels: str,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default NER template for zero-shot annotation

        labels (str): comma-separated list of labels to pass to the template.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match: If False, allow one substring to match multiple times in the text. If True, returns the first hit.

        RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
        """
        self._normalizer = normalizer if normalizer else noop_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in labels.split(",")
        }
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
                text=doc.text, labels=list(self._label_dict.values())
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
            doc.set_ents(filter_spans(spans))
            yield doc
