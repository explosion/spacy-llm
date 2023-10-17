from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, NTokenEstimator, Scorer
from ...ty import TaskResponseParser
from ..util.tokenization import make_default_n_token_estimator
from .parser import parse_responses_v1
from .task import DEFAULT_LEMMA_TEMPLATE_V1, LemmaTask
from .util import LemmaExample, score


@registry.llm_misc("spacy.LemmaParser.v1")
def make_lemma_parser() -> TaskResponseParser[LemmaTask]:
    return parse_responses_v1


@registry.llm_misc("spacy.LemmaScorer.v1")
def make_lemma_scorer() -> Scorer:
    return score


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = DEFAULT_LEMMA_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[LemmaTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    n_token_estimator: Optional[NTokenEstimator] = None,
    scorer: Optional[Scorer] = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    n_token_estimator (Optional[NTokenEstimator]): Estimates number of tokens in a string.
    scorer (Optional[Scorer]): Scorer function.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or LemmaExample
    lemma_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=lemma_examples,
        n_token_estimator=n_token_estimator or make_default_n_token_estimator(),
        scorer=scorer or score,
    )
