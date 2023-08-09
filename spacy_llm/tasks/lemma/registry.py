from typing import Optional

from ...registry import registry
from ...ty import ExamplesConfigType, TaskResponseParser
from .task import LemmaTask
from .util import DEFAULT_LEMMA_TEMPLATE_V1, LemmaExample, parse_responses_v1


@registry.llm_misc("spacy.LemmaParser.v1")
def make_lemma_parser():
    return parse_responses_v1


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = DEFAULT_LEMMA_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser] = None,
    examples: ExamplesConfigType = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    """
    raw_examples = examples() if callable(examples) else examples
    lemma_examples = (
        [LemmaExample(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(
        template=template,
        parse_responses=parse_responses if parse_responses else parse_responses_v1,
        examples=lemma_examples,
    )
