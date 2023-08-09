from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Type

import jinja2
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ...ty import FewshotExample, TaskResponseParser
from ..util import ExampleType, SerializableTask
from .util import DEFAULT_LEMMA_TEMPLATE_V1


class LemmaTask(SerializableTask[ExampleType], Generic[ExampleType]):
    def __init__(
        self,
        template: str = DEFAULT_LEMMA_TEMPLATE_V1,
        parse_responses: Optional[TaskResponseParser] = None,
        fewshot_example_type: Optional[Type[FewshotExample]] = None,
        examples: Optional[List[ExampleType]] = None,
    ):
        """Default lemmatization task.

        template (str): Prompt template passed to the model.
        parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
        fewshot_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        """
        self._template = template
        assert parse_responses is not None
        self._parse_responses = parse_responses
        self._prompt_examples = examples or []
        self._fewshot_example_type = fewshot_example_type

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
                assert self._fewshot_example_type
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
        return self._parse_responses(docs, responses)

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores lemmatization accuracy on provided examples.
        examples (Iterable[Example]): Examples to determine score against.
        """
        return Scorer.score_token_attr(examples, "pos")

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _Example(self) -> Type[FewshotExample]:
        assert self._fewshot_example_type
        return self._fewshot_example_type
