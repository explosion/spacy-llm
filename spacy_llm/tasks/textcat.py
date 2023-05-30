from typing import Callable, Dict, List, Iterable, Optional

import jinja2
from pydantic import BaseModel
from spacy.tokens import Doc
from wasabi import msg

from ..registry import lowercase_normalizer, registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template


_DEFAULT_TEXTCAT_TEMPLATE_v1 = read_template("textcat")
_DEFAULT_TEXTCAT_TEMPLATE_v2 = read_template("textcat.v2")


class TextCatExample(BaseModel):
    text: str
    answer: str


@registry.llm_tasks("spacy.TextCat.v1")
def make_textcat_task(
    labels: str,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
) -> "TextCatTask":
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    textcat_examples = (
        [TextCatExample(**eg) for eg in raw_examples] if raw_examples else None
    )
    return TextCatTask(
        labels=labels_list,
        template=_DEFAULT_TEXTCAT_TEMPLATE_v1,
        examples=textcat_examples,
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
    )


@registry.llm_tasks("spacy.TextCat.v2")
def make_textcat_task_v2(
    labels: str,
    template: str = _DEFAULT_TEXTCAT_TEMPLATE_v2,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
) -> "TextCatTask":
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    textcat_examples = (
        [TextCatExample(**eg) for eg in raw_examples] if raw_examples else None
    )
    return TextCatTask(
        labels=labels_list,
        template=template,
        examples=textcat_examples,
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
    )


class TextCatTask:
    def __init__(
        self,
        labels: List[str],
        template: str = _DEFAULT_TEXTCAT_TEMPLATE_v2,
        examples: Optional[List[TextCatExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        exclusive_classes: bool = False,
        allow_none: bool = True,
        verbose: bool = False,
    ):
        """Default TextCat task.

        You can use either binary or multilabel text classification based on the
        labels you provide.

        If a single label is provided, binary classification
        will be used. The label will get a score of `0` or `1` in `doc.cats`.

        If a comma-separated list of labels is provided, multilabel
        classification will be used. The document labels in `doc.cats` will be a
        dictionary of strings and their score.

        Lastly, you can toggle between exclusive or no-exclusive text
        categorization by passing a flag to the `exclusive_classes` parameter.

        labels (str): Comma-separated list of labels to pass to the template. This task
            assumes binary classification if a single label is provided.
        template (str): Prompt template passed to the model.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        exclusive_classes (bool): If True, require the language model to suggest only one
            label per class. This is automatically set when using binary classification.
        allow_none (bool): if True, there might be cases where no label is applicable.
        verbose (bool): If True, show extra information.
        """
        self._template = template
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {self._normalizer(label): label for label in labels}
        self._examples = examples
        # Textcat configuration
        self._use_binary = True if len(self._label_dict) == 1 else False
        self._exclusive_classes = exclusive_classes
        self._allow_none = allow_none
        self._verbose = verbose

        if self._use_binary and not self._exclusive_classes:
            msg.warn(
                "Binary classification should always be exclusive. Setting "
                "the `exclusive_classes` parameter to True."
            )
            self._exclusive_classes = True

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                examples=self._examples,
                exclusive_classes=self._exclusive_classes,
                allow_none=self._allow_none,
            )
            yield prompt

    def _format_response(self, response: str) -> Dict[str, float]:
        """Parse raw string response into a structured format

        The returned dictionary contains the labels mapped to their score.
        """
        categories: Dict[str, float]
        response = response.strip()
        if self._use_binary:
            # Binary classification: We only have one label
            label: str = list(self._label_dict.values())[0]
            score = 1.0 if response.upper() == "POS" else 0.0
            categories = {label: score}
        else:
            # Multilabel classification
            categories = {label: 0.0 for label in self._label_dict.values()}

            pred_labels = response.split(",")
            if self._exclusive_classes and len(pred_labels) > 1:
                # Don't use anything but raise a debug message
                # Don't raise an error. Let user abort if they want to.
                msg.text(
                    f"LLM returned multiple labels for this exclusive task: {pred_labels}.",
                    " Will store an empty label instead.",
                    show=self._verbose,
                )
                pred_labels = []

            for pred in pred_labels:
                if self._normalizer(pred.strip()) in self._label_dict:
                    category = self._label_dict[self._normalizer(pred.strip())]
                    categories[category] = 1.0
        return categories

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            cats = self._format_response(prompt_response)
            doc.cats = cats
            yield doc
