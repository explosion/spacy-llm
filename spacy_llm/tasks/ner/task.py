from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import filter_spans

from ...compat import Literal
from ...ty import CallableScorableProtocol, TaskResponseParserProtocol
from ..span import SpanExample, SpanTask
from ..templates import read_template

DEFAULT_NER_TEMPLATE_V1 = read_template("ner.v1")
DEFAULT_NER_TEMPLATE_V2 = read_template("ner.v2")


class NERTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str,
        parse_responses: TaskResponseParserProtocol,
        fewshot_example_type: Type[SpanExample],
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[SpanExample]],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],
        case_sensitive_matching: bool,
        single_match: bool,
        scorer: CallableScorableProtocol,
    ):
        """Default NER task.

        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive: Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        scorer (BuiltinScorableProtocol): Scorer function.
        """
        super().__init__(
            labels=labels,
            template=template,
            parse_responses=parse_responses,
            fewshot_example_type=fewshot_example_type,
            label_definitions=label_definitions,
            prompt_examples=prompt_examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
        )
        self._scorer = scorer

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
    ) -> None:
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            labels=labels,
            n_prompt_examples=n_prompt_examples,
        )

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        doc.set_ents(filter_spans(spans))

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    def _extract_labels_from_example(self, example: Example) -> List[str]:
        return [ent.label_ for ent in example.reference.ents]
