import warnings
from typing import Any, Callable, Iterable, List, Optional, Type

import jinja2
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample, TaskResponseParserType
from ..templates import read_template
from ..util import SerializableTask

DEFAULT_SUMMARIZATION_TEMPLATE_V1 = read_template("summarization.v1")


class SummarizationTask(SerializableTask):
    def __init__(
        self,
        parse_responses: TaskResponseParserType,
        fewshot_example_type: Type[FewshotExample],
        template: str,
        max_n_words: Optional[int],
        field: str,
        examples: Optional[List[FewshotExample]],
    ):
        """Default summarization task.

        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        max_n_words (Optional[int]): Max. number of words to use in summary.
        field (str): The name of the doc extension in which to store the summary.
        examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        """
        super().__init__(fewshot_example_type)
        self._template = template
        self._parse_responses = parse_responses
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
                    self._fewshot_example_type.generate(eg, field=self._field)
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
        for doc, summary in zip(docs, self._parse_responses(responses)):
            setattr(doc._, self._field, summary)
            yield doc

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]
