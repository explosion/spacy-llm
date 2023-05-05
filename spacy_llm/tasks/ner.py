from typing import Callable, Iterable, Optional, Tuple

import jinja2
import spacy
from spacy.tokens import Doc
from spacy.util import filter_spans


def find_substrings(
    text: str,
    substrings: Iterable[str],
    *,
    case_sensitive: bool = False,
    single_match: bool = False
) -> Iterable[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions
    in a text. The substrings are assumed to be sorted by the order of their
    occurrence in the text."""

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


@spacy.registry.llm_tasks("spacy.NERZeroShot.v1")
def ner_zeroshot_task(
    labels: str,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Tuple[
    Callable[[Iterable[Doc]], Iterable[str]],
    Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]],
]:
    """Default template where English is used for the task instruction

    labels (str): list of labels to pass to the template, as comma-separated list.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function
    RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
    """

    template = """
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

    label_list = [
        normalizer(label) if normalizer else label for label in labels.split(",")
    ]

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(template)
        for doc in docs:
            prompt = _template.render(text=doc.text, labels=label_list)
            yield prompt

    def _format_response(response: str) -> Iterable[Tuple[str, Iterable[str]]]:
        """Parse raw string response into a structured format"""
        output = []
        for line in response.strip().split("\n"):
            # Check if the formatting we want exists
            # <entity label>: ent1, ent2
            if line and ":" in line:
                label, phrases = line.split(":", 1)
                if label in label_list:
                    # Get the phrases / spans for each entity
                    if phrases.strip():
                        _phrases = [p.strip() for p in phrases.strip().split(",")]
                        output.append((label, _phrases))
        return output

    def prompt_parse(
        docs: Iterable[Doc], prompt_responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            spans = []
            for label, phrases in _format_response(prompt_response):
                if label in label_list:
                    # For each phrase, find the substrings in the text
                    # and create a Span
                    offsets = find_substrings(doc.text, phrases)
                    for start, end in offsets:
                        span = doc.char_span(
                            start, end, alignment_mode="contract", label=label
                        )
                        if span is not None:
                            spans.append(span)
            doc.set_ents(filter_spans(spans))
            yield doc

    return prompt_template, prompt_parse
