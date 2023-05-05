from typing import Callable

import spacy


@spacy.registry.llm_normalizers("spacy.NoOp.v1")
def noop_normalizer() -> Callable[[str], str]:
    """Return the labels as-is

    RETURNS (Callable[[str], str])
    """

    def noop(s: str) -> str:
        return s

    return noop


@spacy.registry.llm_normalizers("spacy.Uppercase.v1")
def uppercase_normalizer() -> Callable[[str], str]:
    """Return uppercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def uppercase(s: str) -> str:
        return s.upper()

    return uppercase


@spacy.registry.llm_normalizers("spacy.Lowercase.v1")
def lowercase_normalizer() -> Callable[[str], str]:
    """Return lowercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def lowercase(s: str) -> str:
        return s.lower()

    return lowercase
