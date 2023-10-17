from typing import Callable

from spacy.tokens import Doc

from ...registry import registry
from ...ty import NTokenEstimator, ShardMapper


@registry.llm_misc("spacy.NTokenEstimator.v1")
def make_n_token_estimator() -> NTokenEstimator:
    """Generates Callable estimating the number of tokens in a given string.
    # todo improve default tokenization (allow language code to do tokenization with pretrained spacy model)
    RETURNS (NTokenEstimator): Callable estimating the number of tokens in a given string.
    """

    def count_tokens_by_spaces(value: str) -> int:
        return len(value.split())

    return count_tokens_by_spaces


@registry.llm_misc("spacy.ShardMapper.v1")
def make_shard_mapper() -> ShardMapper:
    """Generates Callable mapping doc to doc shards fitting within context length.
    RETURNS (ShardMapper): Callable mapping doc to doc shards fitting within context length.
    """

    def map_doc_to_shards(
        doc: Doc, context_length: int, render_template: Callable[[str], str]
    ):
        # todo this is yet a dummy implementation that will fail for texts with len(text) > context length.
        return [doc]

    return map_doc_to_shards
