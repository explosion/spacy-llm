from typing import Callable, Dict, Iterable, List, Tuple

from spacy.tokens import Doc, Span

from ...tasks.util import find_substrings
from .examples import SpanReason
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
    task: SpanTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[Iterable[List[Span]]]:
    """Parses LLM responses for Span tasks.
    task (SpanTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[List[Span]]]): Parsed spans per shard/response.
    """
    for responses_for_doc, shards_for_doc in zip(responses, shards):
        results_for_doc: List[List[Span]] = []

        for shard, response in zip(shards_for_doc, responses_for_doc):
            spans = []
            for label, phrases in _format_response(
                response, task._normalizer, task._label_dict
            ):
                # For each phrase, find the substrings in the text
                # and create a Span
                offsets = find_substrings(
                    shard.text,
                    phrases,
                    case_sensitive=task._case_sensitive_matching,
                    single_match=task._single_match,
                )
                for start, end in offsets:
                    span = shard.char_span(
                        start, end, alignment_mode=task._alignment_mode, label=label
                    )
                    if span is not None:
                        spans.append(span)

            results_for_doc.append(spans)

        yield results_for_doc


def _extract_span_reasons_cot(task: SpanTask, response: str) -> List[SpanReason]:
    """Parse raw string response into a list of SpanReasons.
    task (SpanTask): Task to extract span reasons for.
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
        norm_label = task.normalizer(span_reason.label)
        if norm_label not in task.label_dict:
            continue
        label = task.label_dict[norm_label]
        span_reason.label = label
        span_reasons.append(span_reason)
    return span_reasons


def _find_spans_cot(
    task: SpanTask, doc: Doc, span_reasons: List[SpanReason]
) -> List[Span]:
    """Find a list of spaCy Spans from a list of SpanReasons
    for a single spaCy Doc
    task (SpanTask): Task to extract span reasons for.
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
            case_sensitive=task.case_sensitive_matching,
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
            alignment_mode=task.alignment_mode,
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
        find_after = span.start_char if task.allow_overlap else span.end_char
        prev_span = span
        idx += 1

    return sorted(set(spans))


def parse_responses_cot(
    task: SpanTask, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
) -> Iterable[Iterable[List[Span]]]:
    """Since we provide entities in a numbered list, we expect the LLM to
    output entities in the order they occur in the text. This parse
    function now incrementally finds substrings in the text and tracks the
    last found span's start character to ensure we don't overwrite
    previously found spans.
    task (SpanTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[List[Span]]]): Spans to assign per shard.
    """
    for responses_for_doc, shards_for_doc in zip(responses, shards):
        results_for_doc: List[List[Span]] = []

        for shard, response in zip(shards_for_doc, responses_for_doc):
            span_reasons = _extract_span_reasons_cot(task, response)
            results_for_doc.append(_find_spans_cot(task, shard, span_reasons))

        yield results_for_doc
