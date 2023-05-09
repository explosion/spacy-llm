from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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


@registry.llm_tasks("spacy.NER.v1")
def ner_zeroshot_task(
    labels: str,
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: str = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
) -> Tuple[
    Callable[[Iterable[Doc]], Iterable[str]],
    Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]],
]:
    """Default NER template for LLM annotation

    labels (str): comma-separated list of labels to pass to the template.
    examples (Optional[Callable[[], Iterable[TaskExample]]]): a Callable
        that takes in a path and returns a list of task examples. for few-shot learning.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive: Whether to search without case sensitivity.
    single_match: If False, allow one substring to match multiple times in the text. If True, returns the first hit.

    RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
    """

    if not normalizer:
        normalizer = noop_normalizer()

    # ideally, this list should be taken from spaCy, but it's not currently exposed from doc.pyx.
    alignment_modes = ("strict", "contract", "expand")
    if alignment_mode not in alignment_modes:
        raise ValueError(
            f"Unsupported alignment mode '{alignment_mode}'. Supported modes: {', '.join(alignment_modes)}"
        )

    # Get task examples if the user supplied any
    task_examples: Optional[Iterable[Dict[str, Any]]] = examples() if examples else None

    template = """
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
{{ example['text'] }}
'''
{# whitespace #}
{%- for label, substrings in example['entities'].items() -%}
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

    label_dict = {normalizer(label): label for label in labels.split(",")}

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(label_dict.values()),
                examples=task_examples,
            )
            yield prompt

    def _format_response(response: str) -> Iterable[Tuple[str, Iterable[str]]]:
        """Parse raw string response into a structured format"""
        output = []
        assert normalizer is not None
        for line in response.strip().split("\n"):
            # Check if the formatting we want exists
            # <entity label>: ent1, ent2
            if line and ":" in line:
                label, phrases = line.split(":", 1)
                norm_label = normalizer(label)
                if norm_label in label_dict:
                    # Get the phrases / spans for each entity
                    if phrases.strip():
                        _phrases = [p.strip() for p in phrases.strip().split(",")]
                        output.append((label_dict[norm_label], _phrases))
        return output

    def prompt_parse(
        docs: Iterable[Doc], prompt_responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            spans = []
            for label, phrases in _format_response(prompt_response):
                # For each phrase, find the substrings in the text
                # and create a Span
                offsets = find_substrings(
                    doc.text,
                    phrases,
                    case_sensitive=case_sensitive_matching,
                    single_match=single_match,
                )
                for start, end in offsets:
                    span = doc.char_span(
                        start, end, alignment_mode=alignment_mode, label=label
                    )
                    if span is not None:
                        spans.append(span)
            doc.set_ents(filter_spans(spans))
            yield doc

    return prompt_template, prompt_parse
