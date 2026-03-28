from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from pydantic import BaseModel
from spacy.training import Example
from typing_extensions import Self

from ...ty import FewshotExample


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


class SRLExample(FewshotExample):
    text: str
    predicates: List[PredicateItem]
    relations: List[Tuple[PredicateItem, List[RoleItem]]]

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        return hash((self.text,) + tuple(self.predicates))

    def __str__(self):
        preds = ", ".join([p.text for p in self.predicates])
        rels = [
            (p.text, [(r.label, r.role.text) for r in rs]) for p, rs in self.relations
        ]
        return f"Predicates: {preds}\nRelations: {str(rels)}" ""

    @classmethod
    def generate(cls, example: Example, **kwargs) -> Self:
        return cls(
            text=example.reference.text,
            predicates=example.reference._.predicates,
            relations=example.reference._.relations,
        )


def score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """Score SRL accuracy in examples.
    examples (Iterable[Example]): Examples to score.
    RETURNS (Dict[str, Any]): Dict with metric name -> score.
    """
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
