from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import jinja2
from pydantic import BaseModel
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from spacy_llm.tasks.util.serialization import SerializableTask

from ..registry import lowercase_normalizer, registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template

_DEFAULT_TEXTCAT_TEMPLATE_V1 = read_template("textcat")
_DEFAULT_TEXTCAT_TEMPLATE_V2 = read_template("textcat.v2")
_DEFAULT_TEXTCAT_TEMPLATE_V3 = read_template("textcat.v3")


class TextCatExample(BaseModel):
    text: str
    answer: str


@registry.llm_tasks("spacy.TextCat.v1")
def make_textcat_task(
    labels: str = "",
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
) -> "TextCatTask":
    """TextCat.v1 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    labels (str): Comma-separated list of labels to pass to the template.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    exclusive_classes (bool): If True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.
    allow_none (bool): if True, there might be cases where no label is applicable.
    verbose (bool): If True, show extra information.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    textcat_examples = (
        [TextCatExample(**eg) for eg in raw_examples] if raw_examples else None
    )
    return TextCatTask(
        labels=labels_list,
        template=_DEFAULT_TEXTCAT_TEMPLATE_V1,
        prompt_examples=textcat_examples,
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
    )


@registry.llm_tasks("spacy.TextCat.v2")
def make_textcat_task_v2(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_TEXTCAT_TEMPLATE_V2,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
) -> "TextCatTask":
    """TextCat.v2 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    labels (Union[List[str], str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
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
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    textcat_examples = (
        [TextCatExample(**eg) for eg in raw_examples] if raw_examples else None
    )
    return TextCatTask(
        labels=labels_list,
        template=template,
        prompt_examples=textcat_examples,
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
    )


@registry.llm_tasks("spacy.TextCat.v3")
def make_textcat_task_v3(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_TEXTCAT_TEMPLATE_V3,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
    allow_none: bool = True,
    verbose: bool = False,
) -> "TextCatTask":
    """TextCat.v3 task factory.

    You can use either binary or multilabel text classification based on the
    labels you provide.

    If a single label is provided, binary classification
    will be used. The label will get a score of `0` or `1` in `doc.cats`.

    Otherwise, multilabel classification will be used. The document labels
    in `doc.cats` will be a dictionary of strings and their score.

    Lastly, you can toggle between exclusive or no-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    labels (Union[List[str], str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        This task assumes binary classification if a single label is provided.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Optional dict mapping a label to a description of that label.
        These descriptions are added to the prompt to help instruct the LLM on what to extract.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    exclusive_classes (bool): If True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.
    allow_none (bool): if True, there might be cases where no label is applicable.
    verbose (bool): If True, show extra information.
    """

    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    textcat_examples = (
        [TextCatExample(**eg) for eg in raw_examples] if raw_examples else None
    )

    return TextCatTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=textcat_examples,
        normalizer=normalizer,
        exclusive_classes=exclusive_classes,
        allow_none=allow_none,
        verbose=verbose,
    )


class TextCatTask(SerializableTask[TextCatExample]):
    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_TEXTCAT_TEMPLATE_V3,
        label_definitions: Optional[Dict[str, str]] = None,
        prompt_examples: Optional[List[TextCatExample]] = None,
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

        Otherwise, multilabel classification will be used. The document labels
        in `doc.cats` will be a dictionary of strings and their score.

        Lastly, you can toggle between exclusive or no-exclusive text
        categorization by passing a flag to the `exclusive_classes` parameter.

        labels (List[str]): List of labels to pass to the template. This task
            assumes binary classification if a single label is provided.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Optional dict mapping a label to a description of that label.
            These descriptions are added to the prompt to help instruct the LLM on what to extract.
        prompt_examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
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
        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }
        self._label_definitions = label_definitions
        self._prompt_examples = prompt_examples or []
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

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self._label_dict.values())

    @property
    def prompt_template(self) -> str:
        return self._template

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._prompt_examples,
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

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        return Scorer.score_cats(
            examples,
            attr="cats",
            labels=self._label_dict.values(),
            multi_label=not self._exclusive_classes,
        )

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the TextCat task, by auto-discovering labels.

        Labels can be set through, by order of precedence:

        - the `[initialize]` section of the pipeline configuration
        - the `labels` argument supplied to the task factory
        - the labels found in the examples

        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        n_prompt_examples (int): How many prompt examples to infer from the Example objects.
            0 by default. Takes all examples if set to -1.
        """
        if not labels:
            labels = list(self._label_dict.values())
        infer_labels = not labels

        if infer_labels:
            labels = []

        for eg in get_examples():
            if infer_labels:
                for cat in eg.reference.cats.keys():
                    labels.append(cat)
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(self._create_prompt_example(eg))

        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_template",
            "_label_dict",
            "_label_definitions",
            "_use_binary",
            "_exclusive_classes",
            "_allow_none",
            "_verbose",
        ]

    @property
    def _Example(self) -> Type[TextCatExample]:
        return TextCatExample

    def _create_prompt_example(self, example: Example) -> TextCatExample:
        """Create a textcat prompt example from a spaCy example."""
        if self._use_binary:
            answer = (
                "POS"
                if example.reference.cats[list(self._label_dict.values())[0]] == 1.0
                else "NEG"
            )
        else:
            answer = ",".join(
                [
                    label
                    for label, score in example.reference.cats.items()
                    if score == 1.0
                ]
            )

        textcat_example = TextCatExample(
            text=example.reference.text,
            answer=answer,
        )
        return textcat_example
