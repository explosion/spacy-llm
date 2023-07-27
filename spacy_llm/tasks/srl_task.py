from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Any

import re

from collections import defaultdict
import jinja2
from pydantic import BaseModel, ValidationError
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from ..compat import Literal
from ..registry import lowercase_normalizer, registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .templates import read_template
from .util import SerializableTask
from .util.parsing import find_substrings

_DEFAULT_SPAN_SRL_TEMPLATE_V1 = read_template("span-srl.v1")


class SpanItem(BaseModel):
    text: str
    start_char: int
    end_char: int

    def __hash__(self):
        return hash((self.text, self.start_char, self.end_char))


class PredicateItem(SpanItem):
    roleset_id: str = ""

    def __hash__(self):
        return hash((self.text, self.start_char, self.end_char, self.roleset_id))


class ArgRELItem(BaseModel):
    predicate: PredicateItem
    role: SpanItem
    label: str

    def __hash__(self):
        return hash((self.predicate, self.role, self.label))


class SRLExample(BaseModel):
    text: str
    predicates: List[PredicateItem]
    relations: List[ArgRELItem]


def _preannotate(doc: Union[Doc, SRLExample]) -> str:
    """Creates a text version of the document with list of provided predicates."""

    text = doc.text
    preds = ", ".join([s.text for s in doc.predicates])

    formatted_text = f"{text}\nPredicates: {preds}"

    return formatted_text


def score_srl_spans(
    examples: Iterable[Example],
) -> Dict[str, Any]:
    pred_predicates_spans = set()
    gold_predicates_spans = set()

    pred_relation_tuples = set()
    gold_relation_tuples = set()

    for i, eg in enumerate(examples):
        pred_doc = eg.predicted
        gold_doc = eg.reference

        pred_predicates_spans.update(
            [(i, PredicateItem(**dict(p))) for p in pred_doc._.predicates]
        )
        gold_predicates_spans.update(
            [(i, PredicateItem(**dict(p))) for p in gold_doc._.predicates]
        )

        pred_relation_tuples.update(
            [(i, ArgRELItem(**dict(r))) for r in pred_doc._.relations]
        )
        gold_relation_tuples.update(
            [(i, ArgRELItem(**dict(r))) for r in gold_doc._.relations]
        )

    def _overlap_prf(gold: set, pred: set):
        overlap = gold.intersection(pred)
        p = 0.0 if not len(pred) else len(overlap) / len(pred)
        r = 0.0 if not len(gold) else len(overlap) / len(gold)
        f = 0.0 if not p or not r else 2 * p * r / (p + r)
        return p, r, f

    predicates_prf = _overlap_prf(gold_predicates_spans, pred_predicates_spans)
    micro_rel_prf = _overlap_prf(gold_relation_tuples, pred_relation_tuples)

    def _get_label2rels(rel_tuples: Iterable[Tuple[int, ArgRELItem]]):
        label2rels = defaultdict(set)
        for tup in rel_tuples:
            label_ = tup[1].label
            label2rels[label_].add(tup)
        return label2rels

    pred_label2relations = _get_label2rels(pred_relation_tuples)
    gold_label2relations = _get_label2rels(gold_relation_tuples)

    all_labels = set.union(
        set(pred_label2relations.keys()), set(gold_label2relations.keys())
    )
    label2prf = {}
    for label in all_labels:
        pred_label_rels = pred_label2relations[label]
        gold_label_rels = gold_label2relations[label]
        label2prf[label] = _overlap_prf(gold_label_rels, pred_label_rels)

    return {
        "Predicates": predicates_prf,
        "ARGs": {"Overall": micro_rel_prf, "PerLabel": label2prf},
    }


@registry.llm_tasks("spacy.SRL.v1")
def make_srl_task(
    labels: Union[List[str], str] = [],
    template: str = _DEFAULT_SPAN_SRL_TEMPLATE_V1,
    label_definitions: Optional[Dict[str, str]] = None,
    examples: ExamplesConfigType = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = True,
    single_match: bool = True,
    verbose: bool = False,
    predicate_key: str = "Predicate",
):
    """SRL.v1 task factory.

    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    template (str): Prompt template passed to the model.
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (str): "strict", "contract" or "expand".
    case_sensitive_matching: Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    verbose (boole): Verbose ot not
    predicate_key: The str of Predicate in the template
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    rel_examples = [SRLExample(**eg) for eg in raw_examples] if raw_examples else None
    return SRLTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        examples=rel_examples,
        normalizer=normalizer,
        verbose=verbose,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        predicate_key=predicate_key,
    )


class SRLTask(SerializableTask[SRLExample]):
    @property
    def _Example(self) -> Type[SRLExample]:
        return SRLExample

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_verbose",
            "_predicate_key",
            "_alignment_mode",
            "_case_sensitive_matching",
            "_single_match",
        ]

    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_SPAN_SRL_TEMPLATE_V1,
        label_definitions: Optional[Dict[str, str]] = None,
        examples: Optional[List[SRLExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        verbose: bool = False,
        predicate_key: str = "Predicate",
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = True,
        single_match: bool = True,
    ):
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {self._normalizer(label): label for label in labels}
        self._template = template
        self._label_definitions = label_definitions
        self._examples = examples
        self._verbose = verbose
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._single_match = single_match
        self._predicate_key = predicate_key
        self._check_extensions()

    @classmethod
    def _check_extensions(cls):
        """Add `predicates` extension if need be.
        Add `relations`  extension if need be."""

        if not Doc.has_extension("predicates"):
            Doc.set_extension("predicates", default=[])

        if not Doc.has_extension("relations"):
            Doc.set_extension("relations", default=[])

    @staticmethod
    def _validate_alignment(alignment_mode: str):
        """Raises error if specified alignment_mode is not supported.
        alignment_mode (str): Alignment mode to check.
        """
        # ideally, this list should be taken from spaCy, but it's not currently exposed from doc.pyx.
        alignment_modes = ("strict", "contract", "expand")
        if alignment_mode not in alignment_modes:
            raise ValueError(
                f"Unsupported alignment mode '{alignment_mode}'. Supported modes: {', '.join(alignment_modes)}"
            )

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
    ) -> None:
        """Initialize the task, by auto-discovering labels.

        Labels can be set through, by order of precedence:

        - the `[initialize]` section of the pipeline configuration
        - the `labels` argument supplied to the task factory
        - the labels found in the examples

        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        """
        self._check_extensions()

        examples = get_examples()

        if not labels:
            labels = list(self._label_dict.values())

        if not labels:
            label_set = set()

            for eg in examples:
                rels: List[ArgRELItem] = eg.reference._.relations
                for rel in rels:
                    label_set.add(rel.label)
            labels = list(label_set)

        self._label_dict = {self._normalizer(label): label for label in labels}

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
            predicates = None
            if len(doc._.predicates):
                predicates = ", ".join([p["text"] for p in doc._.predicates])

            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                predicates=predicates,
            )

            yield prompt

    def _format_response(self, arg_lines):
        """Parse raw string response into a structured format"""
        output = []
        # this ensures unique arguments in the sentence for a predicate
        found_labels = set()
        for line in arg_lines:
            try:
                if line.strip() and ":" in line:
                    label, phrase = line.strip().split(":", 1)

                    # label is of the form "ARG-n (def)"
                    label = label.split("(")[0].strip()

                    # strip any surrounding quotes
                    phrase = phrase.strip("'\" -")

                    norm_label = self._normalizer(label)
                    if (
                        norm_label in self._label_dict
                        and norm_label not in found_labels
                    ):
                        if phrase.strip():
                            _phrase = phrase.strip()
                            found_labels.add(norm_label)
                            output.append((self._label_dict[norm_label], _phrase))
            except ValidationError:
                msg.warn(
                    "Validation issue",
                    line,
                    show=self._verbose,
                )
        return output

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            predicates = []
            relations = []
            lines = prompt_response.split("\n")

            # match lines that start with {Predicate:, Predicate 1:, Predicate1:}
            # assuming self.predicate_key = "Predicate"
            pred_patt = r"^" + re.escape(self._predicate_key) + r"\b\s*\d*[:\-\s]"
            pred_indices, pred_lines = zip(
                *[
                    (i, line)
                    for i, line in enumerate(lines)
                    if re.search(pred_patt, line)
                ]
            )

            pred_indices = list(pred_indices)

            # extract the predicate strings
            pred_strings = [line.split(":", 1)[1].strip("'\" ") for line in pred_lines]

            # extract the line ranges (s, e) of predicate's content.
            # then extract the pred content lines using the ranges
            pred_indices.append(len(lines))
            pred_ranges = zip(pred_indices[:-1], pred_indices[1:])
            pred_contents = [lines[s:e] for s, e in pred_ranges]

            # assign the spans of the predicates and args
            # then create ArgRELItem from the identified predicates and arguments
            for pred_str, pred_content_lines in zip(pred_strings, pred_contents):
                pred_offsets = list(
                    find_substrings(
                        doc.text, [pred_str], case_sensitive=True, single_match=True
                    )
                )

                # ignore the args if the predicate is not found
                if len(pred_offsets):
                    p_start_char, p_end_char = pred_offsets[0]
                    pred_item = PredicateItem(
                        text=pred_str, start_char=p_start_char, end_char=p_end_char
                    )
                    predicates.append(pred_item.dict())

                    for label, phrase in self._format_response(pred_content_lines):
                        arg_offsets = find_substrings(
                            doc.text,
                            [phrase],
                            case_sensitive=self._case_sensitive_matching,
                            single_match=self._single_match,
                        )
                        for start, end in arg_offsets:
                            arg_item = SpanItem(
                                text=phrase, start_char=start, end_char=end
                            ).dict()
                            arg_rel_item = ArgRELItem(
                                predicate=pred_item, role=arg_item, label=label
                            ).dict()
                            relations.append(arg_rel_item)

            doc._.predicates = predicates
            doc._.relations = relations
            yield doc
