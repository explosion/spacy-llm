from typing import Optional, Type

from ...registry import registry
from ...ty import CallableScorableProtocol, ExamplesConfigType, FewshotExample
from ...ty import TaskResponseParserProtocol
from .examples import LemmaExample
from .parser import parse_responses_v1
from .scorer import score
from .task import DEFAULT_LEMMA_TEMPLATE_V1, LemmaTask


@registry.llm_misc("spacy.LemmaParser.v1")
def make_lemma_parser() -> TaskResponseParserProtocol[LemmaTask]:
    return parse_responses_v1


@registry.llm_misc("spacy.LemmaScorer.v1")
def make_lemma_scorer() -> CallableScorableProtocol:
    return score


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = DEFAULT_LEMMA_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParserProtocol[LemmaTask]] = None,
    fewshot_example_type: Optional[Type[FewshotExample]] = None,
    fewshot_examples: ExamplesConfigType = None,
    scorer: Optional[CallableScorableProtocol] = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    fewshot_examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    scorer (Optional[BuiltinScorableProtocol]): Scorer function.
    """
    raw_examples = (
        fewshot_examples() if callable(fewshot_examples) else fewshot_examples
    )
    example_type = fewshot_example_type or LemmaExample
    lemma_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        fewshot_example_type=example_type,
        fewshot_examples=lemma_examples,
        scorer=scorer or score,
    )
