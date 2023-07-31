from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from spacy.language import Language
from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc, Span
from spacy.training import Example

from ..compat import Literal
from ..registry import registry
from ..ty import RequiredExamplesConfigType
from ..util import split_labels
from .span import SpanExample, SpanReason, SpanTask
from .templates import read_template

_DEFAULT_SPANCAT_TEMPLATE_V3 = read_template("spancat.v3")


@registry.llm_tasks("spacy.SpanCat.v3")
def make_spancat_task_v3(
    examples: RequiredExamplesConfigType,
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_SPANCAT_TEMPLATE_V3,
    description: Optional[str] = None,
    label_definitions: Optional[Dict[str, str]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = False,
):
    """SpanCat.v3 task factory.

    examples (RequiredExamplesConfigType): Callable
        reads a file containing task examples for few-shot learning or inline List of
        dicts to convert into SpanExample instances.
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
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    span_examples = [SpanExample(**eg) for eg in raw_examples]
    if not description:
        description = (
            f"Entities must have one of these labels: {', '.join(labels_list)}."
        )

    return SpanCatTask(
        labels=labels_list,
        template=template,
        description=description,
        label_definitions=label_definitions,
        prompt_examples=span_examples,
        normalizer=normalizer,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
    )


class SpanCatTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str,
        description: str,
        prompt_examples: Optional[List[SpanExample]] = None,
        label_definitions: Optional[Dict[str, str]] = None,
        spans_key: str = "sc",
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal["strict", "contract", "expand"] = "contract",
        case_sensitive_matching: bool = False,
        allow_overlap: Optional[bool] = True,
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
            allow_overlap=allow_overlap,
        )
        self._spans_key = spans_key

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
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
        """
        if not labels:
            labels = list(self._label_dict.values())

        examples = get_examples()
        for eg in examples:
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(self._create_prompt_example(eg))
        if not labels:
            label_set = set()
            for eg in examples:
                for ent in eg.reference.ents:
                    label_set.add(ent.label_)
            labels = sorted(set(label_set))
        self._label_dict = {self._normalizer(label): label for label in labels}

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        doc.spans[self._spans_key] = sorted(spans)

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        return spancat_score(
            examples,
            spans_key=self._spans_key,
            allow_overlap=True,
        )

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_alignment_mode",
            "_case_sensitive_matching",
            "_spans_key",
        ]

    def _create_prompt_example(self, example: Example) -> SpanExample:
        """Create a SpanCat prompt example from a spaCy example."""
        span_reasons = []
        for span in example.reference.spans[self._spans_key]:
            span_reasons.append(
                SpanReason(
                    text=span.text,
                    is_entity=True,
                    label=span.label_,
                    reason=f"is a {span.label_}",
                )
            )
        return SpanExample(
            text=example.reference.text,
            spans=span_reasons,
        )
