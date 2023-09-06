from typing import Callable, Iterable

from ...registry import registry
from .examples import SpanCoTExample, SpanExample
from .task import SpanTask
from .util import check_label_consistency, check_label_consistency_cot


@registry.llm_misc("spacy.LabelCheck.v1")
def make_label_check() -> Callable[[SpanTask], Iterable[SpanExample]]:
    return check_label_consistency


@registry.llm_misc("spacy.LabelCheckCoT.v1")
def make_label_check_cot() -> Callable[[SpanTask], Iterable[SpanCoTExample]]:
    return check_label_consistency_cot
