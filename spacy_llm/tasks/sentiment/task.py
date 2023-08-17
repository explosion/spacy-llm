from typing import Callable, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample, Self, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template
from .util import SentimentExample

DEFAULT_SENTIMENT_TEMPLATE_V1 = read_template("sentiment.v1")


class SentimentTask(BuiltinTask):
    def __init__(
        self,
        template: str,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        field: str,
        prompt_examples: Optional[List[SentimentExample]],
    ):
        """Sentiment analysis task.

        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        field (str): The name of the doc extension in which to store the sentiment score.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
        )
        self._field = field
        self._check_doc_extension()

    def _check_doc_extension(self):
        """Add extension if need be."""
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        """Initialize sentiment task.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        self._check_doc_extension()
        super()._initialize(
            get_examples=get_examples, nlp=nlp, n_prompt_examples=n_prompt_examples
        )

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        return super().generate_prompts(docs=docs)

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_doc_extension()

        for doc, sentiment_score in zip(
            docs, self._parse_responses(self, docs, responses)
        ):
            try:
                setattr(doc._, self._field, sentiment_score)
            except ValueError:
                setattr(doc._, self._field, None)

            yield doc

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def field(self) -> str:
        return self._field
