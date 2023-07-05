from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from spacy.language import Language
from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc, Span
from spacy.training import Example

from ..compat import Literal
from ..registry import registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .span import SpanExample, SpanTask
from .templates import read_template

_DEFAULT_SPANCAT_TEMPLATE_V1 = read_template("spancat")
_DEFAULT_SPANCAT_TEMPLATE_V2 = read_template("spancat.v2")


@registry.llm_tasks("spacy.SpanCat.v1")
def make_spancat_task(
    labels: str = "",
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    """SpanCat.v1 task factory.

    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
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
    labels_list = split_labels(labels)
    span_examples = (
        [SpanExample(**eg) for eg in examples()] if callable(examples) else examples
    )
    return SpanCatTask(
        labels=labels_list,
        template=_DEFAULT_SPANCAT_TEMPLATE_V1,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


@registry.llm_tasks("spacy.SpanCat.v2")
def make_spancat_task_v2(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_SPANCAT_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    """SpanCat.v2 task factory.

    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
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
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    span_examples = [SpanExample(**eg) for eg in raw_examples] if raw_examples else None
    return SpanCatTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


class SpanCatTask(SpanTask):
    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_SPANCAT_TEMPLATE_V2,
        label_definitions: Optional[Dict[str, str]] = None,
        spans_key: str = "sc",
        prompt_examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default SpanCat task.

        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        spans_key (str): Key of the `Doc.spans` dict to save under.
        prompt_examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
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
            prompt_examples=prompt_examples,
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

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the SpanCat task, by auto-discovering labels.

        Labels can be set through, by order of precedence:

        - the `[initialize]` section of the pipeline configuration
        - the `labels` argument supplied to the task factory
        - the labels found in the examples

        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        n_prompt_examples (int): How many prompt examples to infer from the Example objects.
            0 by default. Takes all examples if set to -1.
        """
        if not labels:
            labels = list(self._label_dict.values())
        infer_labels = not labels

        for eg in get_examples():
            if infer_labels:
                for span in eg.reference.spans.get(self._spans_key, []):
                    labels.append(span.label_)
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(self._create_prompt_example(eg))

        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_spans_key",
            "_label_dict",
            "_template",
            "_label_definitions",
            "_alignment_mode",
            "_case_sensitive_matching",
            "_single_match",
        ]

    def _create_prompt_example(self, example: Example) -> SpanExample:
        """Create a spancat prompt example from a spaCy example."""
        entities = defaultdict(list)
        for span in example.reference.spans[self._spans_key]:
            entities[span.label_].append(span.text)

        return SpanExample(text=example.reference.text, entities=entities)
