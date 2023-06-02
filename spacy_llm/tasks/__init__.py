from .ner import NERTask, make_ner_task, make_ner_task_v2
from .noop import NoopTask, make_noop_task
from .rel import RELTask
from .spancat import SpanCatTask, make_spancat_task, make_spancat_task_v2
from .textcat import TextCatTask, make_textcat_task

__all__ = [
    "NoopTask",
    "NERTask",
    "SpanCatTask",
    "TextCatTask",
    "make_ner_task",
    "make_ner_task_v2",
    "make_noop_task",
    "make_spancat_task",
    "make_spancat_task_v2",
    "make_textcat_task",
    "TextCatTask",
    "SpanCatTask",
    "RELTask",
]
