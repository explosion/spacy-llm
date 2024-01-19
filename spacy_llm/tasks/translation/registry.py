from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ..util.sharding import make_shard_mapper
from .parser import parse_responses_v1
from .task import DEFAULT_TRANSLATION_TEMPLATE_V1, TranslationTask
from .util import TranslationExample, reduce_shards_to_doc


@registry.llm_misc("spacy.TranslationShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.Translation.v1")
def make_translation_task(
    target_lang: str,
    source_lang: Optional[str] = None,
    template: str = DEFAULT_TRANSLATION_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[TranslationTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    field: str = "translation",
):
    """Translation.v1 task factory.

    target_lang (str): Language to translate the text to.
    source_lang (Optional[str]): Language the text is in.
    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser[SummarizationTask]]): Callable for parsing LLM responses for
        this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    field (str): The name of the doc extension in which to store the summary.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or TranslationExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return TranslationTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=span_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        field=field,
        source_lang=source_lang,
        target_lang=target_lang,
    )
