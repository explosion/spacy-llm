from typing import Callable, Dict, List, Optional, Type, Union

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, Scorer, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ...util import split_labels
from ..util.sharding import make_shard_mapper
from .parser import parse_responses_v1_v2_v3
from .task import DEFAULT_TEXTCAT_TEMPLATE_V1, DEFAULT_TEXTCAT_TEMPLATE_V2
from .task import DEFAULT_TEXTCAT_TEMPLATE_V3, TextCatTask
from .util import TextCatExample, reduce_shards_to_doc, score


@registry.llm_misc("spacy.TextCatShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.TextCat.v1")
def make_textcat_task(
    parse_responses: Optional[TaskResponseParser[TextCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: str = "",
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
    scorer: Optional[Scorer] = None,
) -> "TextCatTask":
    """TextCat.v1 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    parse_responses (Optional[TaskResponseParser[TextCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (str): Comma-separated list of labels to pass to the template.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    exclusive_classes (bool): If True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.
    allow_none (bool): if True, there might be cases where no label is applicable.
    verbose (bool): If True, show extra information.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or TextCatExample
    textcat_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )
    return TextCatTask(
        parse_responses=parse_responses or parse_responses_v1_v2_v3,
        prompt_example_type=example_type,
        labels=labels_list,
        template=DEFAULT_TEXTCAT_TEMPLATE_V1,
        prompt_examples=textcat_examples,
        shard_mapper=make_shard_mapper(),
        shard_reducer=make_shard_reducer(),
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
        label_definitions=None,
        scorer=scorer or score,
    )


@registry.llm_tasks("spacy.TextCat.v2")
def make_textcat_task_v2(
    parse_responses: Optional[TaskResponseParser[TextCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_TEXTCAT_TEMPLATE_V2,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
    scorer: Optional[Scorer] = None,
) -> "TextCatTask":
    """TextCat.v2 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    parse_responses (Optional[TaskResponseParser[TextCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (Union[List[str], str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    exclusive_classes (bool): If True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.
    allow_none (bool): if True, there might be cases where no label is applicable.
    verbose (bool): If True, show extra information.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or TextCatExample
    textcat_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return TextCatTask(
        parse_responses=parse_responses or parse_responses_v1_v2_v3,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        prompt_examples=textcat_examples,
        shard_mapper=make_shard_mapper(),
        shard_reducer=make_shard_reducer(),
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
        label_definitions=None,
        scorer=scorer or score,
    )


@registry.llm_tasks("spacy.TextCat.v3")
def make_textcat_task_v3(
    parse_responses: Optional[TaskResponseParser[TextCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_TEXTCAT_TEMPLATE_V3,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
    scorer: Optional[Scorer] = None,
) -> "TextCatTask":
    """TextCat.v3 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    parse_responses (Optional[TaskResponseParser[TextCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (Union[List[str], str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Optional dict mapping a label to a description of that label.
        These descriptions are added to the prompt to help instruct the LLM on what to extract.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    exclusive_classes (bool): If True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.
    allow_none (bool): if True, there might be cases where no label is applicable.
    verbose (bool): If True, show extra information.
    scorer (Optional[Scorer]): Scorer function.
    """

    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or TextCatExample
    textcat_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return TextCatTask(
        parse_responses=parse_responses or parse_responses_v1_v2_v3,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=textcat_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
        scorer=scorer or score,
    )
