from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTaskWithLabels
from ..templates import read_template
from .items import RelationItem

DEFAULT_REL_TEMPLATE: str = read_template("rel.v1")


class RELTask(BuiltinTaskWithLabels):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[FewshotExample[Self]]],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        normalizer: Optional[Callable[[str], str]],
        verbose: bool,
    ):
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
            labels=labels,
            label_definitions=label_definitions,
            normalizer=normalizer,
        )
        """Default REL task. Populates a `Doc._.rel` custom attribute.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in
            prompts.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        verbose (bool): Controls the verbosity of the task.
        """
        self._verbose = verbose
        self._field = "rel"

    def _preprocess_docs_for_prompt(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        preprocessed_docs: List[Doc] = []

        for doc in docs:
            preprocessed_docs.append(
                Doc(doc.vocab, words=RELTask._preannotate(doc).split())
            )
            preprocessed_docs[-1].ents = [
                Span(
                    preprocessed_docs[-1],
                    ent.start,
                    ent.end,
                    label=ent.label_,
                    kb_id=ent.kb_id_,
                )
                for ent in doc.ents
            ]

        return preprocessed_docs

    def _get_prompt_data(self, shard: Doc, n_shards: int) -> Dict[str, Any]:
        return {
            "labels": list(self._label_dict.values()),
            "label_definitions": self._label_definitions,
            "preannotate": RELTask._preannotate,
        }

    @staticmethod
    def _preannotate(doc: Union[Doc, FewshotExample]) -> str:
        """Creates a text version of the document with annotated entities."""
        offset = 0
        text = doc.text

        if not hasattr(doc, "ents"):
            raise ValueError(
                "Prompt example type used in RELTask has to expose entities via an .ents attribute."
            )

        for i, ent in enumerate(doc.ents):
            end = ent.end_char
            before, after = text[: end + offset], text[end + offset :]
            annotation = f"[ENT{i}:{ent.label_ if isinstance(doc, Doc) else ent.label}]"
            offset += len(annotation)
            text = f"{before}{annotation}{after}"

        return text

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        self._check_extension(self._field)
        shards_teed = self._tee_2d_iterable(shards, 2)
        for shards_for_doc, rel_items_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, rel_items in zip(shards_for_doc, rel_items_for_doc):
                shard._.rel = rel_items

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

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
