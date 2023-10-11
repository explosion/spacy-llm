# mypy: ignore-errors
import pytest

from spacy_llm.models.rest.palm import palm_bison

from ..compat import has_palm_key


@pytest.mark.external
@pytest.mark.skipif(has_palm_key is False, reason="PaLM API key not available")
@pytest.mark.parametrize("name", ("text-bison-001", "chat-bison-001"))
def test_palm_api_response_is_correct(name: str):
    """Check if we're getting the response from the correct structure"""
    model = palm_bison(name=name)
    prompt = "The number of stars in the universe is"
    num_prompts = 3  # arbitrary number to check multiple inputs
    responses = list(model([prompt] * num_prompts))
    for response in responses:
        assert isinstance(response, str)
    assert len(responses) == 3


@pytest.mark.external
@pytest.mark.skipif(has_palm_key is False, reason="PaLM API key not available")
def test_palm_api_response_n_generations():
    """Test how the model handles more than 1 generation of output

    Users can configure PaLM to return more than one candidate for a single prompt.
    The current model doesn't support that and the implementation only returns
    the very first output.
    """
    candidate_count = 3
    model = palm_bison(config={"candidate_count": candidate_count})

    prompt = "The number of stars in the universe is"
    num_prompts = 3
    responses = list(model([prompt] * num_prompts))
    assert len(responses) == 3
    for response in responses:
        assert isinstance(response, str)


@pytest.mark.external
@pytest.mark.skipif(has_palm_key is False, reason="PaLM API key not available")
def test_palm_api_response_when_error():
    """Ensure graceful handling of error in the PaLM model."""
    # Incorrect config because temperature is in incorrect range [0, 5]
    # c.f. https://developers.generativeai.google/api/rest/generativelanguage/models/generateText
    incorrect_temperature = 1000  # must be between 0 and 1.0
    with pytest.raises(ValueError, match="Request to PaLM API failed:"):
        palm_bison(config={"temperature": incorrect_temperature})


@pytest.mark.external
@pytest.mark.skipif(has_palm_key is False, reason="PaLM API key not available")
def test_palm_error_unsupported_model():
    """Ensure graceful handling of error when model is not supported"""
    incorrect_model = "x-gpt-3.5-turbo"
    with pytest.raises(ValueError, match="Model 'x-gpt-3.5-turbo' is not supported"):
        palm_bison(name=incorrect_model)
