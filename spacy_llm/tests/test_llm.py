import warnings
from typing import Any, Dict, Tuple, Iterable, Callable

import pytest
import spacy
from spacy.tokens import Doc

from ..pipeline import LLMWrapper
from ..registry import registry

from dotenv import load_dotenv

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
        prompt=None,  # type: ignore
    )
    llm.from_bytes(llm.to_bytes())


def test_llm_serialize_disk():
    llm = LLMWrapper(
        template=None,  # type: ignore
        parse=None,  # type: ignore
        prompt=None,  # type: ignore
    )

    with spacy.util.make_tempdir() as tmp_dir:
        llm.to_disk(tmp_dir / "llm")
        llm.from_disk(tmp_dir / "llm")


@pytest.mark.parametrize(
    "config",
    (
        {
            "prompt": "spacy-llm.MiniChainSimple.v1",
            "api": "spacy-llm.MiniChain.v1",
            "backend": "OpenAI",
            "config": {},
        },
        {
            "prompt": "spacy-llm.LangChainSimple.v1",
            "api": "spacy-llm.LangChain.v1",
            "backend": "openai",
            "config": {"temperature": 0.3},
        },
    ),
)
def test_integrations(config: Dict[str, Any]):
    """Test simple runs with all supported integrations."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "prompt": {
                "@llm.prompts": config["prompt"],
                "api": {
                    "@llm.apis": config["api"],
                    "backend": config["backend"],
                    "config": config["config"],
                },
            }
        },
    )
    nlp("This is a test.")


def test_type_checking() -> None:
    """Tests type checking for consistency between functions."""

    @registry.tasks("spacy-llm.TestIncorrect.v1")
    def noop_task_incorrect() -> Tuple[
        Callable[[Iterable[Doc]], Iterable[int]],
        Callable[[Iterable[Doc], Iterable[int]], Iterable[Doc]],
    ]:
        def prompt_template(docs: Iterable[Doc]) -> Iterable[int]:
            return [0] * len(list(docs))

        def prompt_parse(
            docs: Iterable[Doc], prompt_responses: Iterable[int]
        ) -> Iterable[Doc]:
            return docs

        return prompt_template, prompt_parse

    # Ensure default config doesn't raise warnings.
    nlp = spacy.blank("en")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlp.add_pipe("llm")

    nlp = spacy.blank("en")
    with pytest.warns(UserWarning) as record:
        nlp.add_pipe(
            "llm",
            config={"task": {"@llm.tasks": "spacy-llm.TestIncorrect.v1"}},
        )
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "Type returned from `template()` (`typing.Iterable[int]`) doesn't match type "
        "expected by `prompt()` (`typing.Iterable[str]`)."
    )
    assert (
        str(record[1].message)
        == "Type returned from `prompt()` (`typing.Iterable[str]`) doesn't match type "
        "expected by `parse()` (`typing.Iterable[int]`)."
    )
