from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import jinja2
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...registry import lowercase_normalizer
from ...ty import FewshotExample
from ..templates import read_template
from ..util import SerializableTask
from . import RELExample
from .examples import RelationItem

DEFAULT_REL_TEMPLATE: str = read_template("rel.v1")
TaskResponseParserType = Callable[[Iterable[Any], Iterable[Doc], bool], Iterable[Any]]


class RELTask(SerializableTask):
    def __init__(
        self,
        parse_responses: TaskResponseParserType,
        fewshot_example_type: Type[FewshotExample],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[List[FewshotExample]],
        normalizer: Optional[Callable[[str], str]],
        verbose: bool,
    ):
        """Default REL task. Populates a `Doc._.rel` custom attribute.

        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        fewshot_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        template (str): Prompt template passed to the model.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        prompt_examples (Optional[Callable[[], List[RELExample]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        verbose (bool): Controls the verbosity of the task.
        """
        super().__init__(fewshot_example_type)
        self._parse_responses = parse_responses
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }
        self._template = template
        self._label_definitions = label_definitions
        self._prompt_examples = prompt_examples or []
        self._verbose = verbose

    @classmethod
    def _check_rel_extension(cls):
        """Add `rel` extension if need be."""
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default=[])

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
                text=RELTask._preannotate(doc),
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._prompt_examples,
                preannotate=RELTask._preannotate,
            )
            yield prompt

    @staticmethod
    def _preannotate(doc: Union[Doc, RELExample]) -> str:
        """Creates a text version of the document with annotated entities."""
        offset = 0
        text = doc.text

        for i, ent in enumerate(doc.ents):
            end = ent.end_char
            before, after = text[: end + offset], text[end + offset :]
            annotation = f"[ENT{i}:{ent.label}]"
            offset += len(annotation)
            text = f"{before}{annotation}{after}"

        return text

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_rel_extension()

        for doc, rel_items in zip(
            docs, self._parse_responses(responses, docs, self._verbose)
        ):
            doc._.rel = rel_items
            yield doc

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
    ) -> None:
        """Initialize the SpanCat task, by auto-discovering labels.

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
        self._check_rel_extension()

        if not labels:
            labels = list(self._label_dict.values())
        infer_labels = not labels

        if infer_labels:
            labels = []

        for eg in get_examples():
            if infer_labels:
                rels: List[RelationItem] = eg.reference._.rel
                for rel in rels:
                    labels.append(rel.relation)
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                self._prompt_examples.append(self._fewshot_example_type.generate(eg))

        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_verbose",
        ]
