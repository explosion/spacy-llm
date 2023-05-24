from .ner import NERTask, make_ner_task
from .spancat import SpanCatTask, make_spancat_task
from .textcat import TextCatTask, make_textcat_task
from .noop import NoopTask, make_noop_task

__all__ = [
    "NoopTask",
    "NERTask",
    "SpanCatTask",
    "TextCatTask",
    "make_ner_task",
    "make_noop_task",
    "make_spancat_task",
    "make_textcat_task",
]
