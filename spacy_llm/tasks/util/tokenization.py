from ...registry import registry
from ...ty import NTokenEstimator


@registry.llm_misc("spacy.NTokenEstimator.v1")
def make_default_n_token_estimator() -> NTokenEstimator:
    """Generates Callable estimating the number of tokens in a given string.
    # todo improve default tokenization (allow language code to do tokenization with pretrained spacy model)
    RETURNS (NTokenEstimator): Callable estimating the number of tokens in a given string.
    """

    def count_tokens_by_spaces(value: str) -> int:
        return len(value.split())

    return count_tokens_by_spaces
