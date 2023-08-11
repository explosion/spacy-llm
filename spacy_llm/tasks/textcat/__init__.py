from .examples import TextCatExample
from .registry import make_textcat_task, make_textcat_task_v2, make_textcat_task_v3
from .task import TextCatTask

__all__ = [
    "make_textcat_task",
    "make_textcat_task_v2",
    "make_textcat_task_v3",
    "TextCatExample",
    "TextCatTask",
]
