from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from ...compat import Literal
from ...registry import registry
from ...ty import CallableScorableProtocol, ExamplesConfigType, FewshotExample
from ...ty import TaskResponseParserProtocol
from ...util import split_labels
from ..span import parse_responses as parse_span_responses
from .examples import SpanCatExample
from .scorer import score
from .task import DEFAULT_SPANCAT_TEMPLATE_V1, DEFAULT_SPANCAT_TEMPLATE_V2, SpanCatTask


@registry.llm_tasks("spacy.SpanCat.v1")
def make_spancat_task(
    parse_responses: Optional[TaskResponseParserProtocol[SpanCatTask]] = None,
    fewshot_example_type: Optional[Type[FewshotExample]] = None,
    labels: str = "",
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    spans_key: str = "sc",
    scorer: Optional[CallableScorableProtocol] = None,
):
    """SpanCat.v1 task factory.

    parse_responses (Optional[TaskResponseParserProtocol[SpanCatTask]]): Callable for parsing LLM responses for this
        task.
    fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    fewshot_examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching (bool): Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    spans_key (str): Key of the `Doc.spans` dict to save under.
    scorer (Optional[BuiltinScorableProtocol]): Scorer function.
    """
    labels_list = split_labels(labels)
    example_type = fewshot_example_type or SpanCatExample
    span_examples = (
        [example_type(**eg) for eg in examples()] if callable(examples) else examples
    )

    return SpanCatTask(
        labels=labels_list,
        parse_responses=parse_responses or parse_span_responses,
        fewshot_example_type=example_type,
        template=DEFAULT_SPANCAT_TEMPLATE_V1,
        fewshot_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        label_definitions=None,
        spans_key=spans_key,
        scorer=scorer or score,
    )


@registry.llm_tasks("spacy.SpanCat.v2")
def make_spancat_task_v2(
    parse_responses: Optional[TaskResponseParserProtocol[SpanCatTask]] = None,
    fewshot_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_SPANCAT_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    fewshot_examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    spans_key: str = "sc",
    scorer: Optional[CallableScorableProtocol] = None,
):
    """SpanCat.v2 task factory.

    parse_responses (Optional[TaskResponseParserProtocol[SpanCatTask]]): Callable for parsing LLM responses for this
        task.
    fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    fewshot_examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching (bool): Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    spans_key (str): Key of the `Doc.spans` dict to save under.
    scorer (Optional[BuiltinScorableProtocol]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = (
        fewshot_examples() if callable(fewshot_examples) else fewshot_examples
    )
    example_type = fewshot_example_type or SpanCatExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SpanCatTask(
        parse_responses=parse_responses or parse_span_responses,
        fewshot_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        fewshot_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        spans_key=spans_key,
        scorer=scorer or score,
    )
