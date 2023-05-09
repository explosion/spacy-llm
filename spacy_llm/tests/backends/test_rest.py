import pytest
import spacy
import srsly

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.REST.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3, "model": "text-davinci-003"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)
    nlp("This is a test.")


@pytest.mark.parametrize("strict", (False, True))
def test_rest_backend_error_handling(strict: bool):
    """Test error handling for default/minimal REST backend.
    strict (bool): Whether to use strict mode.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {"@llm_tasks": "spacy.NoOp.v1"},
            "backend": {"config": {"model": "x-text-davinci-003"}, "strict": strict},
        },
    )

    if strict:
        with pytest.raises(
            ValueError,
            match="API call failed: {'error': {'message': 'The model `x-text-davinci-003` does not exist', 'type': "
            "'invalid_request_error', 'param': None, 'code': None}}.",
        ):
            nlp("this is a test")
    else:
        response = nlp.get_pipe("llm")._backend(["this is a test"])
        assert len(response) == 1
        response = srsly.json_loads(response[0])
        assert (
            response["error"]["message"]
            == "The model `x-text-davinci-003` does not exist"
        )


def test_retries():
    """Test retry mechanism."""
    # Run with 0 tries.
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {"@llm_tasks": "spacy.NoOp.v1"},
            "backend": {"config": {"model": "text-davinci-003"}, "n_max_tries": 0},
        },
    )
    cache = nlp.get_pipe("llm")._cache
    with pytest.raises(
        ConnectionError,
        match=f"OpenAI API could not be reached within {cache._timeout} seconds in {cache._n_max_tries} attempts. "
        f"Check your network connection and the availability of the OpenAI API.",
    ):
        nlp("this is a test")
