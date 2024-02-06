from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, Scorer, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_LEMMA_TEMPLATE_V1 = read_template("lemma.v1")


class LemmaTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        prompt_examples: Optional[List[FewshotExample[Self]]],
        template: str,
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        scorer: Scorer,
    ):
        """Default lemmatization task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        scorer (Scorer): Scorer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
        )
        self._scorer = scorer

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)
        for shards_for_doc, lemmas_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            updated_shards_for_doc: List[Doc] = []

            for shard, lemmas in zip(shards_for_doc, lemmas_for_doc):
                tokens = [token for token in shard]
                # If numbers of tokens recognized by spaCy and returned by LLM don't match, we don't attempt a partial
                # match.
                if len(tokens) != len(lemmas):
                    updated_shards_for_doc.append(shard)
                    continue

                # Assign lemmas.
                for token, lemma_info in zip(tokens, lemmas):
                    if len(lemma_info) > 0:
                        token.lemma_ = lemma_info[1]

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

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]
