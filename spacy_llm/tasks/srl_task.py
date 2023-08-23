from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Any

import jinja2
import re
import warnings

from collections import defaultdict
from pydantic import BaseModel, ValidationError
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from ..compat import Literal
from ..registry import registry
from ..ty import ExamplesConfigType
from ..util import split_labels
from .span import SpanTask
from .templates import read_template
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


class RoleItem(BaseModel):
    role: SpanItem
    label: str

    def __hash__(self):
        return hash((self.role, self.label))


class SRLExample(BaseModel):
    text: str
    predicates: List[PredicateItem]
    relations: List[Tuple[PredicateItem, List[RoleItem]]]

    def __hash__(self):
        return hash((self.text,) + tuple(self.predicates))

    def __str__(self):
        return f"""Predicates: {', '.join([p.text for p in self.predicates])}
Relations: {str([(p.text, [(r.label, r.role.text) for r in rs]) for p, rs in self.relations])}"""


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
            [
                (i, PredicateItem(**dict(p)), RoleItem(**dict(r)))
                for p, rs in pred_doc._.relations
                for r in rs
            ]
        )
        gold_relation_tuples.update(
            [
                (i, PredicateItem(**dict(p)), RoleItem(**dict(r)))
                for p, rs in gold_doc._.relations
                for r in rs
            ]
        )

    def _overlap_prf(gold: set, pred: set):
        overlap = gold.intersection(pred)
        p = 0.0 if not len(pred) else len(overlap) / len(pred)
        r = 0.0 if not len(gold) else len(overlap) / len(gold)
        f = 0.0 if not p or not r else 2 * p * r / (p + r)
        return p, r, f

    predicates_prf = _overlap_prf(gold_predicates_spans, pred_predicates_spans)
    micro_rel_prf = _overlap_prf(gold_relation_tuples, pred_relation_tuples)

    def _get_label2rels(rel_tuples: Iterable[Tuple[int, PredicateItem, RoleItem]]):
        label2rels = defaultdict(set)
        for tup in rel_tuples:
            label_ = tup[-1].label
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
    alignment_mode (Literal["strict", "contract", "expand"]): How character indices snap to token boundaries.
        Options: "strict" (no snapping), "contract" (span of all tokens completely within the character span),
        "expand" (span of all tokens at least partially covered by the character span).
        Defaults to "strict".
    case_sensitive_matching: Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    verbose (bool): Verbose or not
    predicate_key: The str of Predicate in the template
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    rel_examples = [SRLExample(**eg) for eg in raw_examples] if raw_examples else None
    return SRLTask(
        labels=labels_list,
        template=template,
        label_definitions=label_definitions,
        prompt_examples=rel_examples,
        normalizer=normalizer,
        verbose=verbose,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        predicate_key=predicate_key,
    )


class SRLTask(SpanTask[SRLExample]):
    def __init__(
        self,
        labels: List[str] = [],
        template: str = _DEFAULT_SPAN_SRL_TEMPLATE_V1,
        label_definitions: Optional[Dict[str, str]] = None,
        prompt_examples: Optional[List[SRLExample]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        alignment_mode: Literal[
            "strict", "contract", "expand"  # noqa: F821
        ] = "contract",
        case_sensitive_matching: bool = True,
        single_match: bool = True,
        verbose: bool = False,
        predicate_key: str = "Predicate",
    ):
        super().__init__(
            labels,
            template,
            label_definitions,
            prompt_examples,
            normalizer,
            alignment_mode,
            case_sensitive_matching,
            single_match,
        )
        self._predicate_key = predicate_key
        self._verbose = verbose
        self._check_extensions()

    def _check_label_consistency(self) -> List[SRLExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.
        RETURNS (List[SRLExample]): List of SRLExamples with valid labels.
        """
        assert self._prompt_examples
        srl_examples = [SRLExample(**eg.dict()) for eg in self._prompt_examples]
        example_labels = {
            self._normalizer(r.label): r.label
            for example in srl_examples
            for p, rs in example.relations
            for r in rs
        }
        unspecified_labels = {
            example_labels[key]
            for key in (set(example_labels.keys()) - set(self._label_dict.keys()))
        }
        if not set(example_labels.keys()) <= set(self._label_dict.keys()):
            warnings.warn(
                f"Examples contain labels that are not specified in the task configuration. The latter contains the "
                f"following labels: {sorted(list(set(self._label_dict.values())))}. Labels in examples missing from "
                f"the task configuration: {sorted(list(unspecified_labels))}. Please ensure your label specification "
                f"and example labels are consistent."
            )

        # Return examples without non-declared roles. the roles within a predicate that have undeclared role labels
        # are discarded.
        return [
            example
            for example in [
                SRLExample(
                    text=example.text,
                    predicates=example.predicates,
                    relations=[
                        (
                            p,
                            [
                                r
                                for r in rs
                                if self._normalizer(r.label) in self._label_dict
                            ],
                        )
                        for p, rs in example.relations
                    ],
                )
                for example in srl_examples
            ]
        ]

    @classmethod
    def _check_extensions(cls):
        """Add `predicates` extension if need be.
        Add `relations`  extension if need be."""

        if not Doc.has_extension("predicates"):
            Doc.set_extension("predicates", default=[])

        if not Doc.has_extension("relations"):
            Doc.set_extension("relations", default=[])

    def initialize(
        self,
        get_examples: Callable[[], Iterable["SRLExample"]],
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
                rels: List[Tuple[PredicateItem, List[RoleItem]]] = eg.relations
                for p, rs in rels:
                    for r in rs:
                        label_set.add(r.label)
            labels = list(label_set)

        self._label_dict = {self._normalizer(label): label for label in labels}

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            predicates = None
            if len(doc._.predicates):
                predicates = ", ".join([p["text"] for p in doc._.predicates])

            doc_examples = self._prompt_examples

            # check if there are doc-tailored examples
            if doc.has_extension("egs") and doc._.egs is not None and len(doc._.egs):
                doc_examples = doc._.egs

            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                predicates=predicates,
                examples=doc_examples,
            )

            yield prompt

    def _format_response(self, arg_lines) -> List[Tuple[str, str]]:
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
        """
        Parse LLM response by extracting predicate-arguments blocks from the generate response.
        For example,
        LLM response for doc: "A sentence with multiple predicates (p1, p2)"

        Step 1: Extract the Predicates for the Text
        Predicates: p1, p2

        Step 2: For each Predicate, extract the Semantic Roles in 'Text'
        Text: A sentence with multiple predicates (p1, p2)
        Predicate: p1
        ARG-0: a0_1
        ARG-1: a1_1
        ARG-M-TMP: a_t_1
        ARG-M-LOC: a_l_1

        Predicate: p2
        ARG-0: a0_2
        ARG-1: a1_2
        ARG-M-TMP: a_t_2

        So the steps in the parsing are to first find the text boundaries for the information
        of each predicate. This is done by identifying the lines "Predicate: p1" and "Predicate: p2",
        which gives us the text for each predicate as follows:

        Predicate: p1
        ARG-0: a0_1
        ARG-1: a1_1
        ARG-M-TMP: a_t_1
        ARG-M-LOC: a_l_1

            and,

        Predicate: p2
        ARG-0: a0_2
        ARG-1: a1_2
        ARG-M-TMP: a_t_2

        Once we separate these out, then it is a matter of parsing line by line to extract the predicate
        and its args for each predicate block

        """
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
                    ).dict()
                    predicates.append(pred_item)

                    roles = []

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
                            arg_rel_item = RoleItem(
                                predicate=pred_item, role=arg_item, label=label
                            ).dict()
                            roles.append(arg_rel_item)

                    relations.append((pred_item, roles))

            doc._.predicates = predicates
            doc._.relations = relations
            yield doc

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

    @property
    def _Example(self) -> Type[SRLExample]:
        return SRLExample
