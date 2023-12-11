from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

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
        return [RELTask._preannotate(doc, True) for doc in docs]

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        return {
            "labels": list(self._label_dict.values()),
            "label_definitions": self._label_definitions,
            "preannotate": RELTask._preannotate,
        }

    @staticmethod
    def _preannotate(
        doc: Union[Doc, FewshotExample], return_as_doc: bool = False
    ) -> Union[str, Doc]:
        """Creates a text version of the document with annotated entities.
        doc (Union[Doc, FewshotExample]): Doc to preannotate.
        return_as_doc (bool): Whether to return as doc (by default returned as text).
        """
        words: List[str] = [] if len(doc.ents) else [t.text for t in doc]
        spaces: List[bool] = [] if len(doc.ents) else [t.whitespace_ != "" for t in doc]
        ent_indices: List[Tuple[int, int]] = []

        # Convert RELExample into Doc for easier subsequent processing.
        # todo Solve import cycle so we can expect RELExample here.
        if not isinstance(doc, Doc):
            assert hasattr(doc, "to_doc") and callable(doc.to_doc)
            doc = doc.to_doc()

        if not hasattr(doc, "ents"):
            raise ValueError(
                "Prompt example type used in RELTask has to expose entities via an .ents attribute."
            )

        # Update token information for doc reconstruction.
        last_ent_end = -1
        for i, ent in enumerate(doc.ents):
            annotation = f"[ENT{i}:{ent.label_ if isinstance(doc, Doc) else ent.label}]"
            tokens_since_last_ent = [
                *[t for t in doc if last_ent_end <= t.i < ent.start],
                *[t for t in ent],
            ]
            words.extend([*[t.text for t in tokens_since_last_ent], annotation])
            spaces.extend([t.whitespace_ != "" for t in tokens_since_last_ent])

            # Adjust spaces w.r.t. added annotations, which should appear directly after entity.
            spaces.append(spaces[-1])
            spaces[-2] = False
            ent_indices.append((ent.start + i, ent.end + i))

            last_ent_end = ent.end

        # Include chars after last ent.
        if len(doc.ents):
            tokens_since_last_ent = [t for t in doc if last_ent_end <= t.i]
            words.extend([t.text for t in tokens_since_last_ent])
            spaces.extend([t.whitespace_ != "" for t in tokens_since_last_ent])

        # Reconstruct doc.
        annotated_doc = Doc(words=words, spaces=spaces, vocab=doc.vocab)
        annotated_doc.ents = [
            Span(  # noqa: E731
                doc=annotated_doc,
                start=ent_idx[0],
                end=ent_idx[1],
                label=doc.ents[i].label,
                vector=doc.ents[i].vector,
                vector_norm=doc.ents[i].vector_norm,
                kb_id=doc.ents[i].kb_id_,
            )
            for i, ent_idx in enumerate(ent_indices)
        ]

        return annotated_doc.text if not return_as_doc else annotated_doc

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
