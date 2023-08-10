from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.pipeline.spancat import spancat_score
from spacy.tokens import Doc, Span
from spacy.training import Example

from ...compat import Literal
from ...ty import TaskResponseParserProtocol
from ..span import SpanExample, SpanTask
from ..templates import read_template

DEFAULT_SPANCAT_TEMPLATE_V1 = read_template("spancat.v1")
DEFAULT_SPANCAT_TEMPLATE_V2 = read_template("spancat.v2")


class SpanCatTask(SpanTask):
    def __init__(
        self,
        parse_responses: TaskResponseParserProtocol,
        fewshot_example_type: Type[SpanExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        spans_key: str,
        prompt_examples: Optional[List[SpanExample]],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],
        case_sensitive_matching: bool,
        single_match: bool,
    ):
        """Default SpanCat task.

        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
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
            parse_responses=parse_responses,
            fewshot_example_type=fewshot_example_type,
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
                self._prompt_examples.append(
                    self._fewshot_example_type.generate(eg, spans_key=self._spans_key)
                )

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
