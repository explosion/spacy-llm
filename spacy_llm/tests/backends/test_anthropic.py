# mypy: ignore-errors
import pytest
from spacy_llm.backends.rest.anthropic import AnthropicBackend

from ..compat import has_anthropic_key


@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_api_response_is_correct():
    """Check if we're getting the expected response and we're parsing it properly"""
    anthropic = AnthropicBackend(
        config={"model": "claude-instant-1", "max_tokens_to_sample": 10},
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


@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_api_response_when_error():
    """Check if error message shows up properly given incorrect config"""
    # Incorrect config c.f. https://console.anthropic.com/docs/api/reference
    incorrect_temperature = "one"  # should be an int
    anthropic = AnthropicBackend(
        config={
            "model": "claude-instant-1",
            "max_tokens_to_sample": 10,
            "temperature": incorrect_temperature,
        },
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )

    prompt = "Count the number of characters in this string: hello"
    with pytest.raises(ValueError) as err:
        anthropic(prompts=[prompt])
    assert "invalid_request_error" in str(err.value)
    assert "temperature: value is not a valid float" in str(err.value)


@pytest.mark.skipif(has_anthropic_key is False, reason="Anthropic API key unavailable")
def test_anthropic_error_unsupported_model():
    """Ensure graceful handling of error when model is not supported"""
    incorrect_model = "x-gpt-3.5-turbo"
    with pytest.raises(ValueError) as err:
        AnthropicBackend(
            config={"model": incorrect_model, "max_tokens_to_sample": 10},
            strict=False,
            max_tries=10,
            interval=5.0,
            max_request_time=20,
        )
    assert "The specified model 'x-gpt-3.5-turbo' is not supported" in str(err.value)
