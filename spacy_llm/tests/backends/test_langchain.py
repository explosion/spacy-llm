import warnings

import spacy
import pytest
from sqlalchemy.exc import MovedIn20Warning

from spacy_llm.compat import has_langchain

PIPE_CFG = {
    "backend": {
        "@llm_backends": "spacy.LangChain.v1",
        "api": "OpenAI",
        "config": {"temperature": 0.3},
        "query": {"@llm_queries": "spacy.CallLangChain.v1"},
    },
    "task": {"@llm_tasks": "spacy.NoOp.v1"},
}


@pytest.mark.external
@pytest.mark.skipif(has_langchain is False, reason="LangChain is not installed")
@pytest.mark.filterwarnings("ignore:^.*pkg_resources.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.MovedIn20Warning")
def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    nlp.add_pipe("llm", config=PIPE_CFG)

    warnings.warn("blub pkg_resources", DeprecationWarning)
    warnings.warn(
        "The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0)",
        MovedIn20Warning,
    )
    nlp("This is a test.")
