# mypy: ignore-errors
import pytest

from spacy_llm.models.rest.cohere import Cohere, Endpoints

from ..compat import has_cohere_key


@pytest.mark.external
@pytest.mark.skipif(has_cohere_key is False, reason="Cohere API key not available")
def test_cohere_api_response_is_correct():
    """Check if we're getting the response from the correct structure"""
    cohere = Cohere(
        name="command",
        endpoint=Endpoints.COMPLETION.value,
        config={},
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )
    prompt = "Count the number of characters in this string: hello"
    num_prompts = 3  # arbitrary number to check multiple inputs
    responses = cohere(prompts=[prompt] * num_prompts)
    for response in responses:
        assert isinstance(response, str)


@pytest.mark.external
@pytest.mark.skipif(has_cohere_key is False, reason="Cohere API key not available")
def test_cohere_api_response_n_generations():
    """Test how the model handles more than 1 generation of output

    Users can configure Cohere to return more than 1 output for a single prompt
    The current model doesn't support that and the implementation only returns
    the very first output.
    """
    num_generations = 3
    cohere = Cohere(
        name="command",
        endpoint=Endpoints.COMPLETION.value,
        config={"num_generations": num_generations},
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )

    prompt = "Count the number of characters in this string: hello"
    num_prompts = 3
    responses = cohere(prompts=[prompt] * num_prompts)
    for response in responses:
        assert isinstance(response, str)


@pytest.mark.external
@pytest.mark.skipif(has_cohere_key is False, reason="Cohere API key not available")
def test_cohere_api_response_when_error():
    """Ensure graceful handling of error in the Cohere model"""
    # Incorrect config because temperature is in incorrect range [0, 5]
    # c.f. https://docs.cohere.com/reference/generate
    incorrect_temperature = 1000  # must be between 0 and 5
    with pytest.raises(ValueError, match="Request to Cohere API failed:"):
        Cohere(
            name="command",
            endpoint=Endpoints.COMPLETION.value,
            config={"temperature": incorrect_temperature},
            strict=False,
            max_tries=10,
            interval=5.0,
            max_request_time=20,
        )


@pytest.mark.external
@pytest.mark.skipif(has_cohere_key is False, reason="Cohere API key not available")
def test_cohere_error_unsupported_model():
    """Ensure graceful handling of error when model is not supported"""
    incorrect_model = "x-gpt-3.5-turbo"
    with pytest.raises(ValueError, match="Model 'x-gpt-3.5-turbo' is not supported"):
        Cohere(
            name=incorrect_model,
            config={},
            endpoint=Endpoints.COMPLETION.value,
            strict=False,
            max_tries=10,
            interval=5.0,
            max_request_time=20,
        )
