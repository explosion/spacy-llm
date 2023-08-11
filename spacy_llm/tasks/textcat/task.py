from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import jinja2
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from spacy_llm.tasks.util.serialization import SerializableTask

from ...registry import lowercase_normalizer
from ...ty import FewshotExample, TaskResponseParserProtocol
from ..templates import read_template

DEFAULT_TEXTCAT_TEMPLATE_V1 = read_template("textcat.v1")
DEFAULT_TEXTCAT_TEMPLATE_V2 = read_template("textcat.v2")
DEFAULT_TEXTCAT_TEMPLATE_V3 = read_template("textcat.v3")


class TextCatTask(SerializableTask):
    def __init__(
        self,
        parse_responses: TaskResponseParserProtocol,
        fewshot_example_type: Type[FewshotExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[FewshotExample]],
        normalizer: Optional[Callable[[str], str]],
        exclusive_classes: bool,
        allow_none: bool,
        verbose: bool,
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

        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
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
        super().__init__(fewshot_example_type)
        self._template = template
        self._parse_responses = parse_responses
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

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, cats in zip(
            docs,
            self._parse_responses(
                responses,
                use_binary=self._use_binary,
                label_dict=self._label_dict,
                normalizer=self._normalizer,
                exclusive_classes=self._exclusive_classes,
                verbose=self._verbose,
            ),
        ):
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
                self._prompt_examples.append(
                    self._fewshot_example_type.generate(
                        eg, use_binary=self._use_binary, label_dict=self._label_dict
                    )
                )

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
