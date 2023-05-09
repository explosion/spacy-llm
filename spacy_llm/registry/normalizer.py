from typing import Callable

from .util import registry


@registry.misc("spacy.NoopNormalizer.v1")
def noop_normalizer() -> Callable[[str], str]:
    """Return the labels as-is

    RETURNS (Callable[[str], str])
    """

    def noop(s: str) -> str:
        return s

    return noop


@registry.misc("spacy.LowercaseNormalizer.v1")
def lowercase_normalizer() -> Callable[[str], str]:
    """Return lowercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def lowercase(s: str) -> str:
        return s.lower()

    return lowercase
