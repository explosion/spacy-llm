import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_SUMMARIZATION_TEMPLATE_V1 = read_template("summarization.v1")


class SummarizationTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        template: str,
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        max_n_words: Optional[int],
        field: str,
        prompt_examples: Optional[List[FewshotExample[Self]]],
    ):
        """Default summarization task.

        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        max_n_words (Optional[int]): Max. number of words to use in summary.
        field (str): The name of the doc extension in which to store the summary.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
        )
        self._max_n_words = max_n_words
        self._field = field
        self._check_example_summaries = True

        if not Doc.has_extension(field):
            Doc.set_extension(field, default=None)

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            n_prompt_examples=n_prompt_examples,
        )

    def _check_prompt_example_summary_len(self) -> None:
        """Checks whether summaries of prompt examples are of expected lengths. Warns if they aren't."""
        if self._max_n_words is None:
            return

        for pr_ex in self._prompt_examples:
            len_summary = len(pr_ex.summary.split())
            len_text = len(pr_ex.text.split())
            if len_summary >= len_text * 1.2:
                warnings.warn(
                    f"The provided example '{pr_ex.text[:30]}...' has a summary of token length {len_summary} and a text "
                    f"of token length {len_text}. Ensure that your examples' summaries are shorter than their original "
                    f"texts."
                )
            if len_summary > self._max_n_words * 1.2:
                warnings.warn(
                    f"The provided example '{pr_ex.text[:20]}...' has a summary of length {len_summary}, but "
                    f"`max_n_words` == {self._max_n_words}. If your examples are longer than they should be, the "
                    f"LLM will likely produce responses that are too long."
                )

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        if self._check_example_summaries:
            self._check_prompt_example_summary_len()
            self._check_example_summaries = False

        return {
            "max_n_words": int(self._max_n_words / n_shards)
            if self._max_n_words is not None
            else None
        }

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)

        for shards_for_doc, summaries_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, summary in zip(shards_for_doc, summaries_for_doc):
                setattr(shard._, self._field, summary)

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def field(self) -> str:
        return self._field

    @property
    def max_n_words(self) -> Optional[int]:
        return self._max_n_words
