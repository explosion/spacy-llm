from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from spacy.scorer import get_ner_prf
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import filter_spans

from ..compat import Literal
from ..registry import registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template
from .util import SpanExample, SpanTask

_DEFAULT_NER_TEMPLATE_V1 = read_template("ner")
_DEFAULT_NER_TEMPLATE_V2 = read_template("ner.v2")


@registry.llm_tasks("spacy.NER.v1")
def make_ner_task(
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
    return NERTask(
        labels=labels_list,
        template=_DEFAULT_NER_TEMPLATE_V1,
        examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


@registry.llm_tasks("spacy.NER.v2")
def make_ner_task_v2(
    labels: Union[List[str], str],
    template: str = _DEFAULT_NER_TEMPLATE_V2,
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
    return NERTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


class NERTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str = _DEFAULT_NER_TEMPLATE_V2,
        label_definitions: Optional[Dict[str, str]] = None,
        examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default NER task.

        labels (str): Comma-separated list of labels to pass to the template.
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        """
        super().__init__(
            labels=labels,
            template=template,
            label_definitions=label_definitions,
            examples=examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        doc.set_ents(filter_spans(spans))

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        return get_ner_prf(examples)
