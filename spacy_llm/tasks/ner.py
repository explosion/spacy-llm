from typing import Callable, List, Optional

from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from .templates import read_template
from .util import SpanTask, SpanExample
from ..compat import Literal
from ..registry import registry
from ..ty import ExamplesConfigType
from ..util import split_labels


_DEFAULT_NER_TEMPLATE = read_template("ner")


@registry.llm_tasks("spacy.NER.v1")
def make_ner_task(
    labels: str,
    template: str = _DEFAULT_NER_TEMPLATE,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",  # noqa: F821
    case_sensitive_matching: bool = False,
    single_match: bool = False,
) -> "NERTask":
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    ner_examples = [SpanExample(**eg) for eg in raw_examples] if raw_examples else None
    return NERTask(
        labels=labels_list,
        template=template,
        examples=ner_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


class NERTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str = _DEFAULT_NER_TEMPLATE,
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
