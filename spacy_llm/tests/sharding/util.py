import warnings
from typing import Iterable, List, Optional

from spacy.tokens import Doc
from spacy.training import Example

from spacy_llm.compat import Self
from spacy_llm.registry import registry
from spacy_llm.tasks import BuiltinTask
from spacy_llm.tasks.util.sharding import make_shard_mapper
from spacy_llm.ty import FewshotExample, ShardReducer


def parse_responses(
    task: "ShardingCountTask",
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[Iterable[int]]:
    for responses_for_doc, shards_for_doc in zip(responses, shards):
        results_for_doc: List[int] = []
        for response, shard in zip(responses_for_doc, shards_for_doc):
            results_for_doc.append(int(response))

        yield results_for_doc


def reduce_shards_to_doc(task: "ShardingCountExample", shards: Iterable[Doc]) -> Doc:
    shards = list(shards)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Skipping unsupported user data",
        )
        doc = Doc.from_docs(shards, ensure_whitespace=True)
    doc.user_data["count"] = sum([shard.user_data["count"] for shard in shards])
    return doc


class ShardingCountExample(FewshotExample):
    @classmethod
    def generate(cls, example: Example, task: "ShardingCountTask") -> Optional[Self]:
        return None


@registry.llm_tasks("spacy.CountWithSharding.v1")
class ShardingCountTask(BuiltinTask):
    _PROMPT_TEMPLATE = (
        "Reply with the number of words in this string (and nothing else): '{{ text }}'"
    )

    def __init__(self):
        assert isinstance(reduce_shards_to_doc, ShardReducer)
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=ShardingCountExample,
            template=self._PROMPT_TEMPLATE,
            prompt_examples=[],
            shard_mapper=make_shard_mapper(),
            shard_reducer=reduce_shards_to_doc,
        )

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)

        for shards_for_doc, counts_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, count in zip(shards_for_doc, counts_for_doc):
                shard.user_data["count"] = count

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

    @property
    def prompt_template(self) -> str:
        return self._PROMPT_TEMPLATE

    @property
    def _cfg_keys(self) -> List[str]:
        return []
