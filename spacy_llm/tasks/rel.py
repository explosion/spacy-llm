from typing import Callable, Dict, Iterable, List, Optional

import jinja2
from pydantic import BaseModel, ValidationError, validator
from spacy.tokens import Doc
from wasabi import msg

from ..registry import lowercase_normalizer, registry
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


class RELExample(BaseModel):
    text: str
    relations: List[RelationItem]


def _preannotate(doc: Doc) -> str:
    """Creates a version of the document with annotated entities."""
    offset = 0

    text = doc.text

    for i, ent in enumerate(doc.ents):
        end = ent.end_char
        before, after = text[: end + offset], text[end + offset :]

        annotation = f"[ENT{i}:{ent.label_}]"
        offset += len(annotation)

        text = f"{before}{annotation}{after}"

    return text


@registry.llm_tasks("spacy.REL.v1")
class RELTask:
    """Simple REL task."""

    _TEMPLATE_STR = read_template("rel")

    def __init__(
        self,
        labels: str,
        label_definitions: Optional[Dict[str, str]] = None,
        examples: Optional[Callable[[], Iterable[Dict]]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        verbose: bool = False,
    ):

        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default=[])

        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in labels.split(",")
        }
        self._label_definitions = label_definitions
        self._examples = [RELExample(**eg) for eg in examples()] if examples else None

        self._verbose = verbose

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._TEMPLATE_STR)
        for doc in docs:
            prompt = _template.render(
                text=_preannotate(doc),
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                examples=self._examples,
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
        for doc, prompt_response in zip(docs, responses):
            rels = self._format_response(prompt_response)
            doc._.rel = rels
            yield doc
