from .ner import NERTask
from .noop import NoopTask
from .rel import RELTask
from .spancat import SpanCatTask
from .textcat import TextCatTask

__all__ = [
    "NoopTask",
    "NERTask",
    "TextCatTask",
    "SpanCatTask",
    "RELTask",
]
