import warnings
from typing import Any, Callable, Iterable, List, Optional, Type

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

_DEFAULT_SUMMARIZATION_TEMPLATE_V1 = read_template("summarization")


class SummarizationExample(BaseModel):
    text: str
    summary: str


@registry.llm_tasks("spacy.Summarization.v1")
def make_summarization_task(
    template: str = _DEFAULT_SUMMARIZATION_TEMPLATE_V1,
    examples: ExamplesConfigType = None,
    max_n_words: Optional[int] = None,
    field: str = "summary",
):
    """Summarization.v1 task factory.

    template (str): Prompt template passed to the model.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    max_n_words (int): Max. number of words to use in summary.
    field (str): The name of the doc extension in which to store the summary.
    """
    raw_examples = examples() if callable(examples) else examples
    span_examples = (
        [SummarizationExample(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SummarizationTask(
        template=template, examples=span_examples, max_n_words=max_n_words, field=field
    )


class SummarizationTask(SerializableTask[SummarizationExample]):
    def __init__(
        self,
        template: str = _DEFAULT_SUMMARIZATION_TEMPLATE_V1,
        examples: Optional[List[SummarizationExample]] = None,
        max_n_words: Optional[int] = None,
        field: str = "summary",
    ):
        """Default summarization task.

        template (str): Prompt template passed to the model.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        max_n_words (int): Max. number of words to use in summary.
        field (str): The name of the doc extension in which to store the summary.
        """
        self._template = template
        self._examples = examples
        self._max_n_words = max_n_words
        self._field = field
        self._prompt_examples = examples or []
        self._check_example_summaries = True
        if not Doc.has_extension(field):
            Doc.set_extension(field, default=None)

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initializes prompt examples from Doc examples.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        for eg in get_examples():
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(
                    SummarizationExample(
                        text=eg.reference.text,
                        summary=getattr(eg.reference._, self._field),
                    )
                )

    def _check_prompt_example_summary_len(self) -> None:
        """Checks whether summaries of prompt examples are of expected lengths. Warns if they aren't."""
        if self._max_n_words is None:
            return

        for pr_ex in self._prompt_examples:
            len_summary = len(pr_ex.summary.split())
            len_text = len(pr_ex.text.split())
            if len_summary >= len_text * 1.2:
                warnings.warn(
                    f"The provided example '{pr_ex.text[:30]}...' has a summary of token length {len_summary} and a text "
                    f"of token length {len_text}. Ensure that your examples' summaries are shorter than their original "
                    f"texts."
                )
            if len_summary > self._max_n_words * 1.2:
                warnings.warn(
                    f"The provided example '{pr_ex.text[:20]}...' has a summary of length {len_summary}, but "
                    f"`max_n_words` == {self._max_n_words}. If your examples are longer than they should be, the "
                    f"LLM will likely produce responses that are too long."
                )

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        if self._check_example_summaries:
            self._check_prompt_example_summary_len()
            self._check_example_summaries = False

        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text, examples=self._examples, max_n_words=self._max_n_words
            )
            yield prompt

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            setattr(doc._, self._field, prompt_response.replace("'''", "").strip())
            yield doc

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _Example(self) -> Type[ExampleType]:
        return SummarizationExample
