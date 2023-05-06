from typing import Callable

import spacy


@spacy.registry.misc("spacy.UppercaseNormalizer.v1")
def uppercase_normalizer() -> Callable[[str], str]:
    """Return uppercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def uppercase(s: str) -> str:
        return s.upper()

    return uppercase


@spacy.registry.misc("spacy.LowercaseNormalizer.v1")
def lowercase_normalizer() -> Callable[[str], str]:
    """Return lowercase versions of the labels

    RETURNS (Callable[[str], str])
    """

    def lowercase(s: str) -> str:
        return s.lower()

    return lowercase
