from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, Scorer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template

DEFAULT_LEMMA_TEMPLATE_V1 = read_template("lemma.v1")


class LemmaTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        prompt_examples: Optional[List[FewshotExample]],
        template: str,
        scorer: Scorer,
    ):
        """Default lemmatization task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        scorer (Scorer): Scorer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
        )
        self._scorer = scorer

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, lemmas in zip(docs, self._parse_responses(self, docs, responses)):
            tokens = [token for token in doc]
            # If numbers of tokens recognized by spaCy and returned by LLM don't match, we don't attempt a partial
            # match.
            if len(tokens) != len(lemmas):
                yield doc

            # Assign lemmas.
            for token, lemma_info in zip(tokens, lemmas):
                if len(lemma_info) > 0:
                    token.lemma_ = lemma_info[1]

            yield doc

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        super()._initialize(
            get_examples=get_examples, nlp=nlp, n_prompt_examples=n_prompt_examples
        )

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]
