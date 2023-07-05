from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import jinja2
from pydantic import BaseModel
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ..registry import registry
from ..ty import ExamplesConfigType
from .templates import read_template
from .util import SerializableTask
from .util.serialization import ExampleType

_DEFAULT_SENTIMENT_TEMPLATE_V1 = read_template("sentiment")


class SentimentExample(BaseModel):
    text: str
    score: float


@registry.llm_tasks("spacy.Sentiment.v1")
def make_sentiment_task(
    template: str = _DEFAULT_SENTIMENT_TEMPLATE_V1,
    examples: ExamplesConfigType = None,
):
    """Sentiment.v1 task factory.

    template (str): Prompt template passed to the model.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    """
    raw_examples = examples() if callable(examples) else examples
    sentiment_examples = (
        [SentimentExample(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SentimentTask(template=template, examples=sentiment_examples)


class SentimentTask(SerializableTask[SentimentExample]):
    def __init__(
        self,
        template: str = _DEFAULT_SENTIMENT_TEMPLATE_V1,
        examples: Optional[List[SentimentExample]] = None,
    ):
        """Sentiment analysis task.

        template (str): Prompt template passed to the model.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        """
        self._template = template
        self._examples = examples
        self._prompt_examples = examples or []

    @classmethod
    def _check_doc_extension(cls):
        """Add `sentiment` extension if need be."""
        if not Doc.has_extension("sentiment"):
            Doc.set_extension("sentiment", default=None)

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize sentiment task.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        self._check_doc_extension()

        for eg in get_examples():
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(
                    SentimentExample(
                        text=eg.reference.text, score=eg.reference._.sentiment
                    )
                )

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                examples=self._examples,
            )
            yield prompt

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_doc_extension()

        for doc, prompt_response in zip(docs, responses):
            try:
                doc._.sentiment = float(
                    "".join(prompt_response.replace("Answer:", "").strip().split())
                )
            except ValueError:
                doc._.sentiment = None

            yield doc

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores lemmatization accuracy on provided examples.
        examples (Iterable[Example]): Examples to determine score against.
        """
        # todo
        return {"sentiment_diff": 0}

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _Example(self) -> Type[ExampleType]:
        return SentimentExample
