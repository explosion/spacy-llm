# mypy: ignore-errors
import pytest

from spacy_llm.models.rest.minimax import Endpoints, MiniMax, minimax_v1

from ..compat import has_minimax_key


@pytest.mark.external
@pytest.mark.skipif(
    has_minimax_key is False, reason="MiniMax API key not available"
)
@pytest.mark.parametrize(
    "name", ("MiniMax-M2.5", "MiniMax-M2.5-highspeed")
)
def test_minimax_api_response_is_correct(name: str):
    """Check if we're getting the expected response and parsing it properly"""
    model = minimax_v1(name=name, config={"temperature": 0.0})
    prompt = "Count the number of characters in this string: hello"
    num_prompts = 3
    responses = model(prompts=[[prompt]] * num_prompts)
    for response in responses:
        assert isinstance(response, list)
        assert len(response) == 1
        assert isinstance(response[0], str)


@pytest.mark.external
@pytest.mark.skipif(
    has_minimax_key is False, reason="MiniMax API key not available"
)
def test_minimax_api_response_when_error():
    """Check if error message shows up properly given incorrect config"""
    incorrect_temperature = "one"  # should be a float
    with pytest.raises(ValueError, match="Request to MiniMax API failed:"):
        minimax_v1(
            name="MiniMax-M2.5",
            config={"temperature": incorrect_temperature},
        )


@pytest.mark.external
@pytest.mark.skipif(
    has_minimax_key is False, reason="MiniMax API key not available"
)
def test_minimax_error_unsupported_model():
    """Ensure graceful handling of error when model is not supported"""
    incorrect_model = "x-nonexistent-model"
    with pytest.raises(ValueError, match="Request to MiniMax API failed:"):
        minimax_v1(name=incorrect_model)


def test_minimax_context_lengths():
    """Verify context length definitions for MiniMax models"""
    ctx = MiniMax._get_context_lengths()
    assert "MiniMax-M2.5" in ctx
    assert "MiniMax-M2.5-highspeed" in ctx
    assert "MiniMax-M2.7" in ctx
    assert "MiniMax-M2.7-highspeed" in ctx
    assert ctx["MiniMax-M2.5"] == 204800
    assert ctx["MiniMax-M2.7"] == 1048576


def test_minimax_endpoints():
    """Verify endpoint definitions"""
    assert Endpoints.CHAT.value == "https://api.minimax.io/v1/chat/completions"
