from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ..util.sharding import make_shard_mapper
from .parser import parse_responses_v1
from .task import DEFAULT_SUMMARIZATION_TEMPLATE_V1, SummarizationTask
from .util import SummarizationExample, reduce_shards_to_doc


@registry.llm_misc("spacy.SummarizationShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.Summarization.v1")
def make_summarization_task(
    template: str = DEFAULT_SUMMARIZATION_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[SummarizationTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    max_n_words: Optional[int] = None,
    field: str = "summary",
):
    """Summarization.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser[SummarizationTask]]): Callable for parsing LLM responses for
        this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    max_n_words (int): Max. number of words to use in summary.
    field (str): The name of the doc extension in which to store the summary.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or SummarizationExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SummarizationTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=span_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        max_n_words=max_n_words,
        field=field,
    )
