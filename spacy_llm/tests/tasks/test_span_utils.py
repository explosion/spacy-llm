import pytest

from spacy_llm.tasks.span import SpanReason

SPAN_REASON_FROM_STR_TEST_CASES = {
    "invalid_order_no_error": (
        "1. Golden State Warriors | BASKETBALL_TEAM | is a basketball team in the NBA | True",
        SpanReason(
            text="Golden State Warriors",
            is_entity=False,
            label="is a basketball team in the NBA",
            reason="True",
        ),
    ),
    "invalid_number_of_components": (
        "1. Golden State Warriors | BASKETBALL_TEAM | OTHER SECTION | MORE THINGS | is a basketball team in the NBA | True",
        ValueError(),
    ),
    "valid_entity_numbered": (
        "1. Golden State Warriors | True | BASKETBALL_TEAM | is a basketball team in the NBA",
        SpanReason(
            text="Golden State Warriors",
            is_entity=True,
            label="BASKETBALL_TEAM",
            reason="is a basketball team in the NBA",
        ),
    ),
    "valid_entity_unnumbered": (
        "Golden State Warriors | True | BASKETBALL_TEAM | is a basketball team in the NBA",
        SpanReason(
            text="Golden State Warriors",
            is_entity=True,
            label="BASKETBALL_TEAM",
            reason="is a basketball team in the NBA",
        ),
    ),
}


@pytest.mark.parametrize(
    "response, expected",
    SPAN_REASON_FROM_STR_TEST_CASES.values(),
    ids=SPAN_REASON_FROM_STR_TEST_CASES.keys(),
)
def test_span_reason_from_str(response: str, expected: SpanReason):
    try:
        span_reason = SpanReason.from_str(response)
    except ValueError:
        assert isinstance(expected, ValueError)
    else:
        assert span_reason == expected
