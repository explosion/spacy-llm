from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from ...compat import Literal
from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, Scorer, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ...util import split_labels
from ..span import parse_responses as parse_span_responses
from ..span import parse_responses_cot as parse_span_responses_cot
from ..span.util import check_label_consistency as check_labels
from ..span.util import check_label_consistency_cot as check_labels_cot
from ..util.sharding import make_shard_mapper
from .task import DEFAULT_SPANCAT_TEMPLATE_V1, DEFAULT_SPANCAT_TEMPLATE_V2
from .task import DEFAULT_SPANCAT_TEMPLATE_V3, SpanCatTask
from .util import SpanCatCoTExample, SpanCatExample, reduce_shards_to_doc, score


@registry.llm_misc("spacy.SpanCatShardReducer.v1")
def make_shard_reducer() -> ShardReducer:
    return reduce_shards_to_doc


@registry.llm_tasks("spacy.SpanCat.v1")
def make_spancat_task(
    parse_responses: Optional[TaskResponseParser[SpanCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: str = "",
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    spans_key: str = "sc",
    scorer: Optional[Scorer] = None,
):
    """SpanCat.v1 task factory.

    parse_responses (Optional[TaskResponseParser[SpanCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching (bool): Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    spans_key (str): Key of the `Doc.spans` dict to save under.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    example_type = prompt_example_type or SpanCatExample
    span_examples = (
        [example_type(**eg) for eg in examples()] if callable(examples) else examples
    )

    return SpanCatTask(
        labels=labels_list,
        parse_responses=parse_responses or parse_span_responses,
        prompt_example_type=example_type,
        template=DEFAULT_SPANCAT_TEMPLATE_V1,
        prompt_examples=span_examples,
        shard_mapper=make_shard_mapper(),
        shard_reducer=make_shard_reducer(),
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        label_definitions=None,
        spans_key=spans_key,
        scorer=scorer or score,
        description=None,
        check_label_consistency=check_labels,
    )


@registry.llm_tasks("spacy.SpanCat.v2")
def make_spancat_task_v2(
    parse_responses: Optional[TaskResponseParser[SpanCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_SPANCAT_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    spans_key: str = "sc",
    scorer: Optional[Scorer] = None,
):
    """SpanCat.v2 task factory.

    parse_responses (Optional[TaskResponseParser[SpanCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching (bool): Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    spans_key (str): Key of the `Doc.spans` dict to save under.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or SpanCatExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SpanCatTask(
        parse_responses=parse_responses or parse_span_responses,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        shard_mapper=make_shard_mapper(),
        shard_reducer=make_shard_reducer(),
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        spans_key=spans_key,
        scorer=scorer or score,
        description=None,
        check_label_consistency=check_labels,
    )


@registry.llm_tasks("spacy.SpanCat.v3")
def make_spancat_task_v3(
    parse_responses: Optional[TaskResponseParser[SpanCatTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_SPANCAT_TEMPLATE_V3,
    description: Optional[str] = None,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    shard_mapper: Optional[ShardMapper] = None,
    shard_reducer: Optional[ShardReducer] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    spans_key: str = "sc",
    scorer: Optional[Scorer] = None,
):
    """SpanCat.v3 task factory for SpanCat with chain-of-thought prompting.

    parse_responses (Optional[TaskResponseParser[SpanCatTask]]): Callable for parsing LLM responses for this
        task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    description (str): A description of what to recognize or not recognize as entities.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    shard_mapper (Optional[ShardMapper]): Maps docs to shards if they don't fit into the model context.
    shard_reducer (Optional[ShardReducer]): Reduces doc shards back into one doc instance.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching (bool): Whether to search without case sensitivity.
    spans_key (str): Key of the `Doc.spans` dict to save under.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or SpanCatCoTExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SpanCatTask(
        parse_responses=parse_responses or parse_span_responses_cot,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        shard_mapper=shard_mapper or make_shard_mapper(),
        shard_reducer=shard_reducer or make_shard_reducer(),
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=False,
        spans_key=spans_key,
        scorer=scorer or score,
        description=description,
        check_label_consistency=check_labels_cot,
    )
