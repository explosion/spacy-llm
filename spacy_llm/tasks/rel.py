from typing import Callable, Dict, Iterable, List, Optional, Union

import jinja2
from pydantic import BaseModel, Field, ValidationError, validator
from spacy.tokens import Doc
from wasabi import msg

from ..registry import lowercase_normalizer, registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template


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
    labels: Union[List[str], str],
    template: str = _DEFAULT_REL_TEMPLATE,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    verbose: bool = False,
) -> "RELTask":
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    rel_examples = [RELExample(**eg) for eg in raw_examples] if raw_examples else None
    return RELTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        examples=rel_examples,
        normalizer=normalizer,
        verbose=verbose,
    )


class RELTask:
    """Simple REL task. Populates a `Doc._.rel` custom attribute."""

    def __init__(
        self,
        labels: List[str],
        template: str = _DEFAULT_REL_TEMPLATE,
        label_definitions: Optional[Dict[str, str]] = None,
        examples: Optional[List[RELExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        verbose: bool = False,
    ):
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {self._normalizer(label): label for label in labels}
        self._template = template
        self._label_definitions = label_definitions
        self._examples = examples
        self._verbose = verbose

    @classmethod
    def _check_rel_extention(cls):
        """Add `rel` extension if need be."""
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default=[])

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=_preannotate(doc),
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._examples,
                preannotate=_preannotate,
            )
            yield prompt

    def _format_response(self, response: str) -> Iterable[RelationItem]:
        """Parse raw string response into a structured format"""
        relations = []
        for line in response.strip().split("\n"):
            try:
                relations.append(RelationItem.parse_raw(line))
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
        self._check_rel_extention()

        for doc, prompt_response in zip(docs, responses):
            rels = self._format_response(prompt_response)
            doc._.rel = rels
            yield doc
