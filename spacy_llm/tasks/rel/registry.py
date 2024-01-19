from typing import Callable, Dict, List, Optional, Type, Union

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ...util import split_labels
from ..util.sharding import make_shard_mapper
from .parser import parse_responses_v1
from .task import DEFAULT_REL_TEMPLATE, RELTask
from .util import RELExample, reduce_shards_to_doc


@registry.llm_misc("spacy.RELShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.REL.v1")
def make_rel_task(
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_REL_TEMPLATE,
    parse_responses: Optional[TaskResponseParser[RELTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    verbose: bool = False,
) -> "RELTask":
    """REL.v1 task factory.

    The REL task populates a `Doc._.rel` custom attribute.

    labels (List[str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser[RELTask]]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    verbose (bool): Controls the verbosity of the task.
    """
    labels_list = split_labels(labels)
    example_type = prompt_example_type or RELExample
    raw_examples = examples() if callable(examples) else examples
    rel_examples = [example_type(**eg) for eg in raw_examples] if raw_examples else None

    return RELTask(
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=rel_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        normalizer=normalizer,
        verbose=verbose,
    )
