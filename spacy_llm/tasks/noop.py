from typing import Iterable, Optional, Tuple

from spacy.tokens import Doc

from ..registry import registry

_NOOP_PROMPT = "Don't do anything."


@registry.llm_tasks("spacy.NoOp.v1")
def make_noop_task():
    return NoopTask()


class NoopTask:
    def generate_prompts(
        self, docs: Iterable[Doc], context_length: Optional[int] = None
    ) -> Iterable[Tuple[Iterable[str], Iterable[Doc]]]:
        for doc in docs:
            yield [_NOOP_PROMPT], [doc]

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        # Grab the first shard per doc
        return [list(shards_for_doc)[0] for shards_for_doc in shards]

    @property
    def prompt_template(self) -> str:
        return """
        This is the NoOp
        prompt template
        """
