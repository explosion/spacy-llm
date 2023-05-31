from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc, Span
from spacy.training import Example

from ..compat import Literal
from ..registry import registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template
from .util import SpanExample, SpanTask

_DEFAULT_SPANCAT_TEMPLATE_V1 = read_template("spancat")
_DEFAULT_SPANCAT_TEMPLATE_V2 = read_template("spancat.v2")


@registry.llm_tasks("spacy.SpanCat.v1")
def make_spancat_task(
    labels: str,
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",  # noqa: F821
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    labels_list = split_labels(labels)
    span_examples = (
        [SpanExample(**eg) for eg in examples()] if callable(examples) else examples
    )
    return SpanCatTask(
        labels=labels_list,
        template=_DEFAULT_SPANCAT_TEMPLATE_V1,
        examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


@registry.llm_tasks("spacy.SpanCat.v2")
def make_spancat_task_v2(
    labels: Union[List[str], str],
    template: str = _DEFAULT_SPANCAT_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",  # noqa: F821
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    span_examples = [SpanExample(**eg) for eg in raw_examples] if raw_examples else None
    return SpanCatTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


class SpanCatTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str = _DEFAULT_SPANCAT_TEMPLATE_V2,
        label_definitions: Optional[Dict[str, str]] = None,
        spans_key: str = "sc",
        examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default SpanCat task.

        labels (str): Comma-separated list of labels to pass to the template.
        spans_key (str): Key of the `Doc.spans` dict to save under.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        """
        super(SpanCatTask, self).__init__(
            labels=labels,
            template=template,
            label_definitions=label_definitions,
            examples=examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )
        self._spans_key = spans_key

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        doc.spans[self._spans_key] = sorted(spans)  # type: ignore [type-var]

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        return spancat_score(
            examples,
            spans_key=self._spans_key,
            allow_overlap=True,
        )
