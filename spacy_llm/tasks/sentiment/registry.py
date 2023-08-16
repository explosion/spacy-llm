from typing import Optional, Type

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, TaskResponseParserProtocol
from .examples import SentimentExample
from .parser import parse_responses_v1
from .task import DEFAULT_SENTIMENT_TEMPLATE_V1, SentimentTask


@registry.llm_tasks("spacy.Sentiment.v1")
def make_sentiment_task(
    template: str = DEFAULT_SENTIMENT_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParserProtocol[SentimentTask]] = None,
    fewshot_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    field: str = "sentiment",
):
    """Sentiment.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParserProtocol[SentimentTask]]): Callable for parsing LLM responses for this task.
    fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    field (str): The name of the doc extension in which to store the summary.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = fewshot_example_type or SentimentExample
    sentiment_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SentimentTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        fewshot_example_type=example_type,
        examples=sentiment_examples,
        field=field,
    )