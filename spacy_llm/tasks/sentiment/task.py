from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample, Scorer, Self, ShardMapper, ShardReducer
from ...ty import TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_SENTIMENT_TEMPLATE_V1 = read_template("sentiment.v1")


class SentimentTask(BuiltinTask):
    def __init__(
        self,
        template: str,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        field: str,
        prompt_examples: Optional[List[FewshotExample[Self]]],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        scorer: Scorer,
    ):
        """Sentiment analysis task.

        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        field (str): The name of the doc extension in which to store the sentiment score.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in
            prompts.
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
        self._scorer = scorer
        self._check_doc_extension()

    def _check_doc_extension(self):
        """Add extension if need be."""
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        """Initialize sentiment task.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        self._check_doc_extension()
        super()._initialize(
            get_examples=get_examples, nlp=nlp, n_prompt_examples=n_prompt_examples
        )

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        self._check_doc_extension()
        shards_teed = self._tee_2d_iterable(shards, 2)

        for shards_for_doc, scores_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, score in zip(shards_for_doc, scores_for_doc):
                try:
                    setattr(shard._, self._field, score)
                except ValueError:
                    setattr(shard._, self._field, None)

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples, field=self._field)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def field(self) -> str:
        return self._field
