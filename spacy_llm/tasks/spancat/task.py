from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example

from ...compat import Literal, Self
from ...ty import FewshotExample, Scorer, ShardMapper, ShardReducer, TaskResponseParser
from ..span import SpanTask
from ..span.task import SpanTaskLabelCheck
from ..templates import read_template

DEFAULT_SPANCAT_TEMPLATE_V1 = read_template("spancat.v1")
DEFAULT_SPANCAT_TEMPLATE_V2 = read_template("spancat.v2")
DEFAULT_SPANCAT_TEMPLATE_V3 = read_template("spancat.v3")


class SpanCatTask(SpanTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        spans_key: str,
        prompt_examples: Optional[List[FewshotExample[Self]]],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],
        case_sensitive_matching: bool,
        single_match: bool,
        scorer: Scorer,
        description: Optional[str],
        check_label_consistency: SpanTaskLabelCheck[Self],
    ):
        """Default SpanCat task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        spans_key (str): Key of the `Doc.spans` dict to save under.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive_matching (bool): Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        scorer (Scorer): Scorer function.
        description (str): A description of what to recognize or not recognize as entities.
        check_label_consistency (SpanTaskLabelCheck): Callable to check label consistency.
        """
        super(SpanCatTask, self).__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            labels=labels,
            template=template,
            label_definitions=label_definitions,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
            description=description,
            allow_overlap=True,
            check_label_consistency=check_label_consistency,
        )
        self._spans_key = spans_key
        self._scorer = scorer

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        doc.spans[self._spans_key] = sorted(spans)  # type: ignore [type-var]

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples, spans_key=self._spans_key, allow_overlap=True)

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
            spans_key=self._spans_key,
        )

    @property
    def _cfg_keys(self) -> List[str]:
        return [*super()._cfg_keys, "_spans_key"]

    def _extract_labels_from_example(self, example: Example) -> List[str]:
        return [
            span.label_ for span in example.reference.spans.get(self._spans_key, [])
        ]

    @property
    def spans_key(self) -> str:
        return self._spans_key
