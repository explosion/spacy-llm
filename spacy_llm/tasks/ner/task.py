from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import filter_spans

from ...compat import Literal, Self
from ...ty import FewshotExample, Scorer, ShardMapper, ShardReducer, TaskResponseParser
from ..span import SpanTask
from ..span.task import SpanTaskLabelCheck
from ..templates import read_template

DEFAULT_NER_TEMPLATE_V1 = read_template("ner.v1")
DEFAULT_NER_TEMPLATE_V2 = read_template("ner.v2")
DEFAULT_NER_TEMPLATE_V3 = read_template("ner.v3")


class NERTask(SpanTask):
    def __init__(
        self,
        labels: List[str],
        template: str,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        label_definitions: Optional[Dict[str, str]],
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
        """Default NER task.

        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser[SpanTask]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive_matching (bool): Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        scorer (Scorer): Scorer function.
        description (str): A description of what to recognize or not recognize as entities.
        check_label_consistency (SpanTaskLabelCheck): Callable to check label consistency.
        """
        super().__init__(
            labels=labels,
            template=template,
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
            label_definitions=label_definitions,
            prompt_examples=prompt_examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
            description=description,
            allow_overlap=False,
            check_label_consistency=check_label_consistency,
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
