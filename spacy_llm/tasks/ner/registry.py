from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from ...compat import Literal
from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, Scorer, TaskResponseParser
from ...util import split_labels
from ..span import parse_responses as parse_span_responses
from .task import DEFAULT_NER_TEMPLATE_V1, DEFAULT_NER_TEMPLATE_V2, NERTask, SpanTask
from .util import NERExample, score


@registry.llm_tasks("spacy.NER.v1")
def make_ner_task(
    parse_responses: Optional[TaskResponseParser[SpanTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: str = "",
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    scorer: Optional[Scorer] = None,
):
    """NER.v1 task factory.

    parse_responses (Optional[TaskResponseParser[SpanTask]]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching: Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    example_type = prompt_example_type or NERExample
    span_examples = (
        [example_type(**eg) for eg in examples()] if callable(examples) else examples
    )

    return NERTask(
        parse_responses=parse_responses or parse_span_responses,
        prompt_example_type=example_type,
        labels=labels_list,
        template=DEFAULT_NER_TEMPLATE_V1,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        label_definitions=None,
        scorer=scorer or score,
    )


@registry.llm_tasks("spacy.NER.v2")
def make_ner_task_v2(
    parse_responses: Optional[TaskResponseParser[SpanTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    labels: Union[List[str], str] = [],
    template: str = DEFAULT_NER_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
    scorer: Optional[Scorer] = None,
):
    """NER.v2 task factory.

    parse_responses (Optional[TaskResponseParser[SpanTask]]): Callable for parsing LLM responses for this task.
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
    scorer (Optional[Scorer]): Scorer function.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or NERExample
    span_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return NERTask(
        parse_responses=parse_responses or parse_span_responses,
        prompt_example_type=example_type,
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        scorer=scorer or score,
    )
