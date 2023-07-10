# mypy: ignore-errors
import re

import pytest

from spacy_llm.models.rest.anthropic import Anthropic, Endpoints

from ..compat import has_anthropic_key


@pytest.mark.external
@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_api_response_is_correct():
    """Check if we're getting the expected response and we're parsing it properly"""
    anthropic = Anthropic(
        name="claude-instant-1",
        endpoint=Endpoints.COMPLETIONS.value,
        config={"max_tokens_to_sample": 10},
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )

    prompt = "Count the number of characters in this string: hello"
    num_prompts = 3
    responses = anthropic(prompts=[prompt] * num_prompts)
    for response in responses:
        assert isinstance(response, str)


@pytest.mark.external
@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_api_response_when_error():
    """Check if error message shows up properly given incorrect config"""
    # Incorrect config c.f. https://console.anthropic.com/docs/api/reference
    incorrect_temperature = "one"  # should be an int
    with pytest.raises(ValueError, match="Request to Anthropic API failed:"):
        Anthropic(
            name="claude-instant-1",
            endpoint=Endpoints.COMPLETIONS.value,
            config={
                "max_tokens_to_sample": 10,
                "temperature": incorrect_temperature,
            },
            strict=False,
            max_tries=10,
            interval=5.0,
            max_request_time=20,
        )


@pytest.mark.external
@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_error_unsupported_model():
    """Ensure graceful handling of error when model is not supported"""
    incorrect_model = "x-gpt-3.5-turbo"
    with pytest.raises(
        ValueError, match=re.escape("Model 'x-gpt-3.5-turbo' is not supported")
    ):
        Anthropic(
            name=incorrect_model,
            endpoint=Endpoints.COMPLETIONS.value,
            config={"max_tokens_to_sample": 10},
            strict=False,
            max_tries=10,
            interval=5.0,
            max_request_time=20,
        )
