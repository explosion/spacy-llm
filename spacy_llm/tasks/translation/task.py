from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_TRANSLATION_TEMPLATE_V1 = read_template("translation.v1")


class TranslationTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        prompt_examples: Optional[List[FewshotExample[Self]]],
        template: str,
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        field: str,
        source_lang: Optional[str],
        target_lang: str,
    ):
        """Default summarization task.

        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        field (str): The name of the doc extension in which to store the summary.
        source_lang (Optional[str]): Language the text is in.
        target_lang (str): Language to translate the text to.
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
        self._source_lang = source_lang
        self._target_lang = target_lang

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

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        return {"source_lang": self._source_lang, "target_lang": self._target_lang}

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)

        for shards_for_doc, translations_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, translation in zip(shards_for_doc, translations_for_doc):
                setattr(shard._, self._field, translation)

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def field(self) -> str:
        return self._field
