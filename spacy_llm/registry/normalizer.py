from typing import Callable

from .util import registry


@registry.misc("spacy.StripNormalizer.v1")
def strip_normalizer() -> Callable[[str], str]:
    """Return the labels as-is with stripped whitespaces

    RETURNS (Callable[[str], str])
    """

    def noop(s: str) -> str:
        return s.strip()

    return noop


@registry.misc("spacy.LowercaseNormalizer.v1")
def lowercase_normalizer() -> Callable[[str], str]:
    """Return lowercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def lowercase(s: str) -> str:
        return s.strip().lower()

    return lowercase
