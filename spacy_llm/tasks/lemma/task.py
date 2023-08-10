from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import jinja2
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample, TaskResponseParserType
from ..templates import read_template
from ..util import SerializableTask

DEFAULT_LEMMA_TEMPLATE_V1 = read_template("lemma.v1")


class LemmaTask(SerializableTask):
    def __init__(
        self,
        parse_responses: TaskResponseParserType,
        fewshot_example_type: Type[FewshotExample],
        template: str = DEFAULT_LEMMA_TEMPLATE_V1,
        examples: Optional[List[FewshotExample]] = None,
    ):
        """Default lemmatization task.
        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        template (str): Prompt template passed to the model.
        examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        """
        super().__init__(fewshot_example_type)
        self._template = template
        self._parse_responses = parse_responses
        self._prompt_examples = examples or []

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
                self._prompt_examples.append(self._fewshot_example_type.generate(eg))

    @property
    def prompt_template(self) -> str:
        return self._template

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                examples=self._prompt_examples,
            )
            yield prompt

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, lemmas in zip(docs, self._parse_responses(responses)):
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

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores lemmatization accuracy on provided examples.
        examples (Iterable[Example]): Examples to determine score against.
        """
        return Scorer.score_token_attr(examples, "lemma")

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _example_type(self) -> Type[FewshotExample]:
        return self._fewshot_example_type
