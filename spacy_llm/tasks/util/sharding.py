from typing import Callable, Iterable, List, Optional, Union

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
def make_shard_mapper(
    n_token_estimator: Optional[NTokenEstimator] = None,
    buffer_frac: float = 1.1,
) -> ShardMapper:
    """Generates Callable mapping doc to doc shards fitting within context length.
    n_token_estimator (NTokenEstimator): Estimates number of tokens in a string.
    buffer_frac (float): Buffer to consider in assessment of whether prompt fits into context. E. g. if value is 1.1,
        prompt length * 1.1 will be compared with the context length.
    todo sharding would be better with sentences instead of tokens, but this requires some form of sentence
     splitting we can't rely one...maybe checking for sentences and/or as optional arg?
    RETURNS (ShardMapper): Callable mapping doc to doc shards fitting within context length.
    """
    n_tok_est: NTokenEstimator = n_token_estimator or make_n_token_estimator()

    def map_doc_to_shards(
        doc: Doc,
        i_doc: int,
        context_length: int,
        render_template: Callable[[Doc, int, int, int], str],
    ) -> Union[Iterable[Doc], Doc]:
        prompt = render_template(doc, 0, i_doc, 1)

        # If prompt with complete doc too long: split in shards.
        if n_tok_est(prompt) * buffer_frac > context_length:
            shards: List[Doc] = []
            # Prompt length unfortunately can't be exacted computed prior to rendering the prompt, as external
            # information not present in the doc (e. g. entity description for EL prompts) may be injected.
            # For this reason we follow a greedy binary search heuristic, if the fully rendered prompt is too long:
            #   1. Get total number of tokens/sentences (depending on the reducer's configuration)
            #   2. Splice off doc up to the first half of tokens/sentences
            #   3. Render prompt and check whether it fits into context
            #   4. If yes: repeat with second doc half.
            #   5. If not: repeat from 2., but with split off shard instead of doc.
            remaining_doc: Optional[Doc] = doc.copy()
            fraction = 0.5
            start_idx = 0
            n_shards = 1

            while remaining_doc is not None:
                fits_in_context = False
                shard: Optional[Doc] = None
                end_idx = -1
                n_tries = 0

                while fits_in_context is False:
                    end_idx = start_idx + int(len(remaining_doc) * fraction)
                    shard = doc[start_idx:end_idx].as_doc(copy_user_data=True)
                    fits_in_context = (
                        n_tok_est(render_template(shard, len(shards), i_doc, n_shards))
                        * buffer_frac
                        <= context_length
                    )
                    fraction /= 2
                    n_tries += 1

                    # If prompt is too large even with shard of a single token, raise error - we can't shard any more
                    # than this. This is an edge case and will most likely never occur.
                    if len(shard) == 1 and not fits_in_context:
                        raise ValueError(
                            "Prompt size doesn't allow for the inclusion for shard of length 1. Please "
                            "review your prompt and reduce its size."
                        )

                assert shard is not None
                shards.append(shard)
                fraction = 1
                n_shards = max(len(shards) + round(1 / fraction), 1)
                start_idx = end_idx
                # Set remaining_doc to None if shard contains all of it, i. e. entire original doc has been processed.
                remaining_doc = (
                    doc[end_idx:].as_doc(copy_user_data=True)
                    if shard.text != remaining_doc.text
                    else None
                )

            return shards

        else:
            return [doc]

    return map_doc_to_shards
