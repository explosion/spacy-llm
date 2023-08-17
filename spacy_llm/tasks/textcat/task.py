from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from ...compat import Self
from ...ty import FewshotExample, Scorer, TaskResponseParser
from ..builtin_task import BuiltinTaskWithLabels
from ..templates import read_template

DEFAULT_TEXTCAT_TEMPLATE_V1 = read_template("textcat.v1")
DEFAULT_TEXTCAT_TEMPLATE_V2 = read_template("textcat.v2")
DEFAULT_TEXTCAT_TEMPLATE_V3 = read_template("textcat.v3")


class TextCatTask(BuiltinTaskWithLabels):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[FewshotExample]],
        normalizer: Optional[Callable[[str], str]],
        exclusive_classes: bool,
        allow_none: bool,
        verbose: bool,
        scorer: Scorer,
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

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        labels (List[str]): List of labels to pass to the template. This task
            assumes binary classification if a single label is provided.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Optional dict mapping a label to a description of that label.
            These descriptions are added to the prompt to help instruct the LLM on what to extract.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        exclusive_classes (bool): If True, require the language model to suggest only one
            label per class. This is automatically set when using binary classification.
        allow_none (bool): if True, there might be cases where no label is applicable.
        verbose (bool): If True, show extra information.
        scorer (Scorer): Scorer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            labels=labels,
            label_definitions=label_definitions,
            normalizer=normalizer,
        )
        # Textcat configuration
        self._use_binary = True if len(self._label_dict) == 1 else False
        self._exclusive_classes = exclusive_classes
        self._allow_none = allow_none
        self._verbose = verbose
        self._scorer = scorer

        if self._use_binary and not self._exclusive_classes:
            msg.warn(
                "Binary classification should always be exclusive. Setting "
                "the `exclusive_classes` parameter to True."
            )
            self._exclusive_classes = True

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        return super().generate_prompts(
            docs=docs,
            labels=list(self._label_dict.values()),
            label_definitions=self._label_definitions,
            exclusive_classes=self._exclusive_classes,
            allow_none=self._allow_none,
        )

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, cats in zip(docs, self._parse_responses(self, docs, responses)):
            doc.cats = cats
            yield doc

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        return self._scorer(
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
    ) -> None:
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            labels=labels,
            n_prompt_examples=n_prompt_examples,
            use_binary=self._use_binary,
            label_dict=self._label_dict,
        )

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

    def _extract_labels_from_example(self, example: Example) -> List[str]:
        return list(example.reference.cats.keys())

    @property
    def use_binary(self) -> bool:
        return self._use_binary

    @property
    def exclusive_classes(self) -> bool:
        return self._exclusive_classes

    @property
    def allow_none(self) -> bool:
        return self._allow_none

    @property
    def verbose(self) -> bool:
        return self._verbose
