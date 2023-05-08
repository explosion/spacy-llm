import warnings
from typing import Any, Callable, Dict, Iterable, Tuple

import pytest
import spacy
import srsly
from dotenv import load_dotenv
from spacy.tokens import Doc

from ..pipeline import LLMWrapper

load_dotenv()  # take environment variables from .env.


@pytest.fixture
def nlp() -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("llm")
    return nlp


def test_llm_init(nlp):
    """Test pipeline intialization."""
    assert ["llm"] == nlp.pipe_names


def test_llm_pipe(nlp):
    """Test call .pipe()."""
    docs = list(nlp.pipe(texts=["This is a test", "This is another test"]))
    assert len(docs) == 2


def test_llm_serialize_bytes():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        backend=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        backend=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


@pytest.mark.parametrize(
    "config",
    (
        {
            "query": "spacy.RunMiniChain.v1",
            "backend": "spacy.MiniChain.v1",
            "api": "OpenAI",
            "config": {},
        },
        {
            "query": "spacy.CallLangChain.v1",
            "backend": "spacy.LangChain.v1",
            "api": "openai",
            "config": {"temperature": 0.3},
        },
        {
            "query": "spacy.CallMinimal.v1",
            "backend": "spacy.Minimal.v1",
            "api": "OpenAI",
            "config": {"temperature": 0.3, "model": "text-davinci-003"},
        },
    ),
)
def test_backends(config: Dict[str, Any]):
    """Test simple runs with all supported backends."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "backend": {
                "api": config["api"],
                "@llm_backends": config["backend"],
                "config": {},
                "query": {"@llm_queries": config["query"]},
            },
        },
    )
    nlp("This is a test.")


def test_type_checking() -> None:
    """Test type checking for consistency between functions."""

    @spacy.registry.llm_tasks("spacy.TestIncorrect.v1")
    def noop_task_incorrect() -> Tuple[
        Callable[[Iterable[Doc]], Iterable[int]],
        Callable[[Iterable[Doc], Iterable[int]], Iterable[Doc]],
    ]:
        def template(docs: Iterable[Doc]) -> Iterable[int]:
            return [0] * len(list(docs))

        def parse(
            docs: Iterable[Doc], prompt_responses: Iterable[int]
        ) -> Iterable[Doc]:
            return docs

        return template, parse

    # Ensure default config doesn't raise warnings.
    nlp = spacy.blank("en")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlp.add_pipe("llm")

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning) as record:
        nlp.add_pipe(
            "llm",
            config={"task": {"@llm_tasks": "spacy.TestIncorrect.v1"}},
        )
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "Type returned from `task[0]` (`typing.Iterable[int]`) doesn't match type "
        "expected by `backend` (`typing.Iterable[str]`)."
    )
    assert (
        str(record[1].message)
        == "Type returned from `backend` (`typing.Iterable[str]`) doesn't match type "
        "expected by `parse` (`typing.Iterable[int]`)."
    )


@pytest.mark.parametrize("strict", (False, True))
def test_minimal_backend_error_handling(strict: bool):
    """Test error handling for default/minimal backend.
    strict (bool): Whether to use strict mode.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "backend": {"config": {"model": "x-text-davinci-003"}, "strict": strict},
        },
    )

    if strict:
        with pytest.raises(ValueError) as error:
            nlp("this is a test")
        assert (
            str(error.value)
            == "API call failed: {'error': {'message': 'The model `x-text-davinci-003` does not "
            "exist', 'type': 'invalid_request_error', 'param': None, 'code': None}}."
        )
    else:
        response = nlp.get_pipe("llm")._backend(["this is a test"])
        assert len(response) == 1
        response = srsly.json_loads(response[0])
        assert (
            response["error"]["message"]
            == "The model `x-text-davinci-003` does not exist"
        )
