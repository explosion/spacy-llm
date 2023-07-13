from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import jinja2
from pydantic import BaseModel, Field, ValidationError, validator
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from ..registry import lowercase_normalizer, registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template
from .util import SerializableTask


class RelationItem(BaseModel):
    dep: int
    dest: int
    relation: str

    @validator("dep", "dest", pre=True)
    def clean_ent(cls, value):
        if isinstance(value, str):
            value = value.strip("ENT")
        return value


class EntityItem(BaseModel):
    start_char: int
    end_char: int
    label_: str = Field(alias="label")


class RELExample(BaseModel):
    text: str
    ents: List[EntityItem]
    relations: List[RelationItem]


def _preannotate(doc: Union[Doc, RELExample]) -> str:
    """Creates a text version of the document with annotated entities."""
    offset = 0

    text = doc.text

    for i, ent in enumerate(doc.ents):
        end = ent.end_char
        before, after = text[: end + offset], text[end + offset :]

        annotation = f"[ENT{i}:{ent.label_}]"
        offset += len(annotation)

        text = f"{before}{annotation}{after}"

    return text


_DEFAULT_REL_TEMPLATE = read_template("rel")


@registry.llm_tasks("spacy.REL.v1")
def make_rel_task(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_REL_TEMPLATE,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    verbose: bool = False,
) -> "RELTask":
    """REL.v1 task factory.

    The REL task populates a `Doc._.rel` custom attribute.

    labels (List[str]): List of labels to pass to the template,
        either an actual list or a comma-separated string.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    examples (Optional[Callable[[], List[RELExample]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
    verbose (bool): Controls the verbosity of the task.
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    rel_examples = [RELExample(**eg) for eg in raw_examples] if raw_examples else None
    return RELTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=rel_examples,
        normalizer=normalizer,
        verbose=verbose,
    )


class RELTask(SerializableTask[RELExample]):
    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_REL_TEMPLATE,
        label_definitions: Optional[Dict[str, str]] = None,
        prompt_examples: Optional[List[RELExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        verbose: bool = False,
    ):
        """Default REL task. Populates a `Doc._.rel` custom attribute.

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
                text=_preannotate(doc),
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._prompt_examples,
                preannotate=_preannotate,
            )
            yield prompt

    def _format_response(self, response: str, doc: Doc) -> List[RelationItem]:
        """Parse raw string response into a structured format"""
        relations = []
        for line in response.strip().split("\n"):
            try:
                rel_item = RelationItem.parse_raw(line)
                if 0 <= rel_item.dep < len(doc.ents) and 0 <= rel_item.dest < len(
                    doc.ents
                ):
                    relations.append(rel_item)
            except ValidationError:
                msg.warn(
                    "Validation issue",
                    line,
                    show=self._verbose,
                )
        return relations

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_rel_extension()

        for doc, prompt_response in zip(docs, responses):
            rels = self._format_response(prompt_response, doc)
            doc._.rel = rels
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
                self._prompt_examples.append(self._create_prompt_example(eg))

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

    @property
    def _Example(self) -> Type[RELExample]:
        return RELExample

    def _create_prompt_example(self, example: Example) -> RELExample:
        """Create a REL prompt example from a spaCy example."""
        entities = [
            EntityItem(
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=ent.label_,
            )
            for ent in example.reference.ents
        ]

        rel_example = RELExample(
            text=example.reference.text,
            ents=entities,
            relations=example.reference._.rel,
        )
        return rel_example
