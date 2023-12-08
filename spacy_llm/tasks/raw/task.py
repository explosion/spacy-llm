from typing import Callable, Iterable, List, Optional, Type

from spacy import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_RAW_TEMPLATE_V1 = read_template("raw.v1")


class RawTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        prompt_examples: Optional[List[FewshotExample[Self]]],
        template: str,
        field: str,
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
    ):
        """Raw task. Expects prompt template without instructions for LLM, i. e. docs have to provide instructions
            themselves.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        field (str): Field to store responses in.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
        )
        self._field = field
        self._check_doc_extension()

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)
        for shards_for_doc, responses_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            updated_shards_for_doc: List[Doc] = []
            for shard, response in zip(shards_for_doc, responses_for_doc):
                setattr(shard._, self._field, response)
                updated_shards_for_doc.append(shard)

            yield self._shard_reducer(self, updated_shards_for_doc)  # type: ignore[arg-type]

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        super()._initialize(
            get_examples=get_examples, nlp=nlp, n_prompt_examples=n_prompt_examples
        )

    def _check_doc_extension(self):
        """Add extension if need be."""
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def field(self) -> str:
        """Return field used to store replies in docs.
        RETURNS (str): Field used to store replies in docs.
        """
        return self._field
