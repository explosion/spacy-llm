from .lemma import LemmaTask, make_lemma_task
from .ner import NERTask, make_ner_task, make_ner_task_v2
from .noop import NoopTask, make_noop_task
from .rel import RELTask, make_rel_task
from .spancat import SpanCatTask, make_spancat_task, make_spancat_task_v2
from .textcat import TextCatTask, make_textcat_task

__all__ = [
    "make_lemma_task",
    "make_ner_task",
    "make_ner_task_v2",
    "make_noop_task",
    "make_rel_task",
    "make_spancat_task",
    "make_spancat_task_v2",
    "make_textcat_task",
    "LemmaTask",
    "NERTask",
    "NoopTask",
    "RELTask",
    "SpanCatTask",
    "TextCatTask",
]
