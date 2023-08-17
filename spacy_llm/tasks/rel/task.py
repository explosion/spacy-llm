from typing import Callable, Dict, Iterable, List, Optional, Type, Union

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, TaskResponseParser
from ..builtin_task import BuiltinTaskWithLabels
from ..templates import read_template
from .util import RelationItem, RELExample

DEFAULT_REL_TEMPLATE: str = read_template("rel.v1")


class RELTask(BuiltinTaskWithLabels):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[FewshotExample]],
        normalizer: Optional[Callable[[str], str]],
        verbose: bool,
    ):
        """Default REL task. Populates a `Doc._.rel` custom attribute.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        verbose (bool): Controls the verbosity of the task.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            labels=labels,
            label_definitions=label_definitions,
            normalizer=normalizer,
        )
        self._verbose = verbose
        self._field = "rel"

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        return super().generate_prompts(
            docs=[
                Doc(doc.vocab, words=RELTask._preannotate(doc).split()) for doc in docs
            ],
            labels=list(self._label_dict.values()),
            label_definitions=self._label_definitions,
            preannotate=RELTask._preannotate,
        )

    @staticmethod
    def _preannotate(doc: Union[Doc, RELExample]) -> str:
        """Creates a text version of the document with annotated entities."""
        offset = 0
        text = doc.text

        for i, ent in enumerate(doc.ents):
            end = ent.end_char
            before, after = text[: end + offset], text[end + offset :]
            annotation = f"[ENT{i}:{ent.label}]"
            offset += len(annotation)
            text = f"{before}{annotation}{after}"

        return text

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_extension(self._field)

        for doc, rel_items in zip(docs, self._parse_responses(self, docs, responses)):
            doc._.rel = rel_items
            yield doc

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
    ) -> None:
        self._check_extension(self._field)
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            labels=labels,
            n_prompt_examples=n_prompt_examples,
        )

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_verbose",
        ]

    def _extract_labels_from_example(self, example: Example) -> List[str]:
        rels: List[RelationItem] = example.reference._.rel
        return [rel.relation for rel in rels]

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def field(self) -> str:
        return self._field
