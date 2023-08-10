from typing import Callable, Dict, Iterable, Tuple

from spacy.tokens import Span

from ...tasks.util import find_substrings


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


def parse_responses_v1_v2(responses: Iterable[str], **kwargs) -> Iterable[Span]:
    """Parses LLM responses for Span tasks.
    responses (Iterable[str]): LLM responses.
    kwargs ([Dict[str, Any]): Additional, mandatory arguments:
        case_sensitive (bool): Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in the text. If True, returns the first
            hit.
        alignment_mode (str): "strict", "contract" or "expand".
        normalizer (Callable[[str], str]): normalizer function.
        label_dict (Dict[str, str]): Mapping of normalized to non-normalized labels.
    RETURNS (Iterable[Span]): Parsed spans per doc/response.
    """
    for doc, prompt_response in zip(kwargs["docs"], responses):
        spans = []
        for label, phrases in _format_response(
            prompt_response, kwargs["normalizer"], kwargs["label_dict"]
        ):
            # For each phrase, find the substrings in the text
            # and create a Span
            offsets = find_substrings(
                doc.text,
                phrases,
                case_sensitive=kwargs["case_sensitive_matching"],
                single_match=kwargs["single_match"],
            )
            for start, end in offsets:
                span = doc.char_span(
                    start, end, alignment_mode=kwargs["alignment_mode"], label=label
                )
                if span is not None:
                    spans.append(span)

        yield spans
