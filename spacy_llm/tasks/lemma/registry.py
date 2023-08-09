from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExampleType, TaskResponseParser
from .examples import LemmaExample
from .parser import parse_responses_v1
from .task import DEFAULT_LEMMA_TEMPLATE_V1, LemmaTask


@registry.llm_misc("spacy.LemmaParser.v1")
def make_lemma_parser() -> TaskResponseParser:
    return parse_responses_v1


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = DEFAULT_LEMMA_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser] = None,
    fewshot_example_type: Optional[Type[FewshotExampleType]] = None,
    examples: ExamplesConfigType = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    """
    raw_examples = examples() if callable(examples) else examples
    parse_responses = parse_responses if parse_responses else parse_responses_v1
    example_type = fewshot_example_type or LemmaExample
    lemma_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(
        template=template,
        parse_responses=parse_responses,
        fewshot_example_type=fewshot_example_type,
        examples=lemma_examples,
    )
