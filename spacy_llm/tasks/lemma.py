from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import jinja2
from pydantic import BaseModel
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from ..registry import registry
from ..ty import ExamplesConfigType
from .templates import read_template
from .util import SerializableTask

_DEFAULT_LEMMA_TEMPLATE_V1 = read_template("lemma")


class LemmaExample(BaseModel):
    text: str
    lemmas: List[Dict[str, str]]


@registry.llm_tasks("spacy.Lemma.v1")
def make_lemma_task(
    template: str = _DEFAULT_LEMMA_TEMPLATE_V1,
    examples: ExamplesConfigType = None,
):
    """Lemma.v1 task factory.

    template (str): Prompt template passed to the model.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    """
    raw_examples = examples() if callable(examples) else examples
    lemma_examples = (
        [LemmaExample(**eg) for eg in raw_examples] if raw_examples else None
    )

    return LemmaTask(template=template, examples=lemma_examples)


class LemmaTask(SerializableTask[LemmaExample]):
    def __init__(
        self,
        template: str = _DEFAULT_LEMMA_TEMPLATE_V1,
        examples: Optional[List[LemmaExample]] = None,
    ):
        """Default lemmatization task.

        template (str): Prompt template passed to the model.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        """
        self._template = template
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
                self._prompt_examples.append(self._create_prompt_example(eg))

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
        for doc, prompt_response in zip(docs, responses):
            parsed_response = [
                [pr_part.strip() for pr_part in pr.split(":")]
                for pr in prompt_response.replace("Lemmatized text:", "")
                .replace("'''", "")
                .strip()
                .split("\n")
            ]
            tokens = [token for token in doc]

            # If numbers of tokens recognized by spaCy and returned by LLM don't match, we don't attempt a partial
            # match.
            if len(tokens) != len(parsed_response):
                yield doc

            # Assign lemmas.
            for token, lemma_info in zip(tokens, parsed_response):
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
        return Scorer.score_token_attr(examples, "pos")

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _Example(self) -> Type[LemmaExample]:
        return LemmaExample

    def _create_prompt_example(self, example: Example) -> LemmaExample:
        """Create a lemma prompt example from a spaCy example."""
        lemma_dict = [{t.text: t.lemma_} for t in example.reference]
        return LemmaExample(text=example.reference.text, lemmas=lemma_dict)
