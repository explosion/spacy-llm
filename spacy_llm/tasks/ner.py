import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from spacy.language import Language
from spacy.scorer import get_ner_prf
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import filter_spans

from ..compat import Literal
from ..registry import registry
from ..ty import COTExamplesConfigType, ExamplesConfigType
from ..util import split_labels
from .span import COTSpanExample, SpanExample, SpanReason, SpanTask
from .templates import read_template

_DEFAULT_NER_TEMPLATE_V1 = read_template("ner")
_DEFAULT_NER_TEMPLATE_V2 = read_template("ner.v2")
_DEFAULT_NER_TEMPLATE_V3 = read_template("ner.v3")


@registry.llm_tasks("spacy.NER.v1")
def make_ner_task(
    labels: str = "",
    examples: Optional[Callable[[], Iterable[Any]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    """NER.v1 task factory.

    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
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
    labels_list = split_labels(labels)

    span_examples = (
        [SpanExample(**eg) for eg in examples()] if callable(examples) else examples
    )
    return NERTask(
        labels=labels_list,
        template=_DEFAULT_NER_TEMPLATE_V1,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


@registry.llm_tasks("spacy.NER.v2")
def make_ner_task_v2(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_NER_TEMPLATE_V2,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    """NER.v2 task factory.

    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
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
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    span_examples = [SpanExample(**eg) for eg in raw_examples] if raw_examples else None

    return NERTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


@registry.llm_tasks("spacy.NER.v3")
def make_ner_task_v3(
    examples: COTExamplesConfigType,
    description: str,
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_NER_TEMPLATE_V3,
    label_definitions: Optional[Dict[str, str]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
    single_match: bool = False,
):
    """NER.v3 task factory.

    examples (Union[Callable[[], Iterable[COTS]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    labels (Union[str, List[str]]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive: Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    span_examples = [COTSpanExample(**eg) for eg in raw_examples]

    return COTNERTask(
        labels=labels_list,
        template=template,
        description=description,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
    )


class NERTask(SpanTask[SpanExample]):
    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_NER_TEMPLATE_V2,
        label_definitions: Optional[Dict[str, str]] = None,
        prompt_examples: Optional[List[SpanExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        """Default NER task.

        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
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
            prompt_examples=prompt_examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )

    def _check_label_consistency(self) -> List[SpanExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.
        RETURNS (List[SpanExample]): List of SpanExamples with valid labels.
        """
        assert self._prompt_examples
        example_labels = {
            self._normalizer(key): key
            for example in self._prompt_examples
            for key in example.entities
        }
        unspecified_labels = {
            example_labels[key]
            for key in (set(example_labels.keys()) - set(self._label_dict.keys()))
        }
        if not set(example_labels.keys()) <= set(self._label_dict.keys()):
            warnings.warn(
                f"Examples contain labels that are not specified in the task configuration. The latter contains the "
                f"following labels: {sorted(list(set(self._label_dict.values())))}. Labels in examples missing from "
                f"the task configuration: {sorted(list(unspecified_labels))}. Please ensure your label specification "
                f"and example labels are consistent."
            )

        # Return examples without non-declared labels. If an example only has undeclared labels, it is discarded.
        return [
            example
            for example in [
                SpanExample(
                    text=example.text,
                    entities={
                        label: entities
                        for label, entities in example.entities.items()
                        if self._normalizer(label) in self._label_dict
                    },
                )
                for example in self._prompt_examples
            ]
            if len(example.entities)
        ]

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the NER task, by auto-discovering labels.

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

        if infer_labels:
            labels = []

        for eg in get_examples():
            if infer_labels:
                for ent in eg.reference.ents:
                    labels.append(ent.label_)
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(self._create_prompt_example(eg))

        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }

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

    @property
    def _Example(self) -> type[SpanExample]:
        return SpanExample

    def _create_prompt_example(self, example: Example) -> SpanExample:
        """Create an NER prompt example from a spaCy example."""
        entities = defaultdict(list)
        for ent in example.reference.ents:
            entities[ent.label_].append(ent.text)

        return SpanExample(text=example.reference.text, entities=entities)


class COTNERTask(SpanTask[COTSpanExample]):
    def __init__(
        self,
        labels: List[str],
        template: str,
        description: Optional[str] = None,
        prompt_examples: Optional[List[COTSpanExample]] = None,
        label_definitions: Optional[Dict[str, str]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        single_match: bool = False,
    ):
        super().__init__(
            labels=labels,
            template=template,
            description=description,
            label_definitions=label_definitions,
            prompt_examples=prompt_examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )

    def _check_label_consistency(self) -> List[SpanExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.
        RETURNS (List[SpanExample]): List of SpanExamples with valid labels.
        """
        assert self._prompt_examples
        # TODO: Implement for new COTSpanExample type
        return self._prompt_examples

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        **kwargs: Any,
    ) -> None:
        """Initialize the NER task, by auto-discovering labels.

        Labels can be set through, by order of precedence:

        - the `[initialize]` section of the pipeline configuration
        - the `labels` argument supplied to the task factory
        - the labels found in the examples

        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        """

        examples = get_examples()

        if not labels:
            labels = list(self._label_dict.values())

        if not labels:
            label_set = set()

            for eg in examples:
                for ent in eg.reference.ents:
                    label_set.add(ent.label_)
            labels = list(label_set)

        self._label_dict = {self._normalizer(label): label for label in labels}

    def _format_response(self, response: str) -> Iterable[Tuple[str, Iterable[str]]]:
        """Parse raw string response into a structured format"""
        output: dict[str, list[str]] = defaultdict(list)
        assert self._normalizer is not None
        for line in response.strip().split("\n"):
            entity = SpanReason.from_str(line)
            if entity:
                norm_label = self._normalizer(entity.label)
                if norm_label not in self._label_dict:
                    continue
                label = self._label_dict[norm_label]
                output[label].append(entity.text)
        return output.items()

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

    @property
    def _Example(self) -> type[COTSpanExample]:
        return COTSpanExample
