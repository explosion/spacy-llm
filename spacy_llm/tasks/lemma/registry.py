from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, Scorer, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ..util.sharding import make_shard_mapper
from .parser import parse_responses_v1
from .task import DEFAULT_LEMMA_TEMPLATE_V1, LemmaTask
from .util import LemmaExample, reduce_shards_to_doc, score


@registry.llm_misc("spacy.LemmaParser.v1")
def make_lemma_parser() -> TaskResponseParser[LemmaTask]:
    return parse_responses_v1


@registry.llm_misc("spacy.LemmaScorer.v1")
def make_lemma_scorer() -> Scorer:
    return score


@registry.llm_misc("spacy.LemmaShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = DEFAULT_LEMMA_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[LemmaTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    scorer: Optional[Scorer] = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    scorer (Optional[Scorer]): Scorer function.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or LemmaExample
    lemma_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=lemma_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        scorer=scorer or score,
    )
