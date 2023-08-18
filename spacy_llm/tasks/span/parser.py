from typing import Callable, Dict, Iterable, List, Tuple

from spacy.tokens import Doc, Span

from ...tasks.util import find_substrings
from .task import SpanTask


def _format_response(
    response: str, normalizer: Callable[[str], str], label_dict: Dict[str, str]
) -> Iterable[Tuple[str, Iterable[str]]]:
    """Parse raw string response into a structured format.
    response (str): LLM response.
    normalizer (Callable[[str], str]): normalizer function.
    label_dict (Dict[str, str]): Mapping of normalized to non-normalized labels.
    RETURNS (Iterable[Tuple[str, Iterable[str]]]): Formatted response.
    """
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


def parse_responses(
    task: SpanTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[List[Span]]:
    """Parses LLM responses for Span tasks.
    task (SpanTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Iterable[Span]): Parsed spans per doc/response.
    """
    for doc, prompt_response in zip(docs, responses):
        spans = []
        for label, phrases in _format_response(
            prompt_response, task._normalizer, task._label_dict
        ):
            # For each phrase, find the substrings in the text
            # and create a Span
            offsets = find_substrings(
                doc.text,
                phrases,
                case_sensitive=task._case_sensitive_matching,
                single_match=task._single_match,
            )
            for start, end in offsets:
                span = doc.char_span(
                    start, end, alignment_mode=task._alignment_mode, label=label
                )
                if span is not None:
                    spans.append(span)

        yield spans
