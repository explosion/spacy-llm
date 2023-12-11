import warnings
from typing import Iterable, Optional, Tuple

from spacy.tokens import Doc

from ..registry import registry

_NOOP_PROMPT = "Don't do anything."


@registry.llm_tasks("spacy.NoOp.v1")
def make_noop_task():
    return ShardingNoopTask()


@registry.llm_tasks("spacy.NoOpNoShards.v1")
def make_noopnoshards_task():
    return NoopTask()


class ShardingNoopTask:
    def generate_prompts(
        self, docs: Iterable[Doc], context_length: Optional[int] = None
    ) -> Iterable[Tuple[Iterable[str], Iterable[Doc]]]:
        for doc in docs:
            yield [_NOOP_PROMPT], [doc]

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Skipping .* while merging docs.",
            )
            docs = [
                Doc.from_docs(list(shards_for_doc), ensure_whitespace=True)
                for shards_for_doc in shards
            ]
        return docs

    @property
    def prompt_template(self) -> str:
        return """
        This is the NoOp
        prompt template
        """


class NoopTask:
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield _NOOP_PROMPT

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        return docs

    @property
    def prompt_template(self) -> str:
        return """
        This is the NoOp
        prompt template
        """
