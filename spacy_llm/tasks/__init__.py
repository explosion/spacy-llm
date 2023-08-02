from .legacy.ner import make_ner_task as make_ner_task_v1
from .legacy.ner import make_ner_task_v2
from .legacy.spancat import make_spancat_task as make_spancat_task_v1
from .legacy.spancat import make_spancat_task_v2
from .lemma import LemmaTask, make_lemma_task
from .ner import NERTask, make_ner_task_v3
from .noop import NoopTask, make_noop_task
from .rel import RELTask, make_rel_task
from .sentiment import SentimentTask, make_sentiment_task
from .spancat import SpanCatTask, make_spancat_task_v3
from .summarization import SummarizationTask, make_summarization_task
from .textcat import TextCatTask, make_textcat_task

__all__ = [
    "make_lemma_task",
    "make_ner_task_v1",
    "make_ner_task_v2",
    "make_ner_task_v3",
    "make_noop_task",
    "make_rel_task",
    "make_sentiment_task",
    "make_spancat_task_v1",
    "make_spancat_task_v2",
    "make_spancat_task_v3",
    "make_summarization_task",
    "make_textcat_task",
    "LemmaTask",
    "NERTask",
    "NoopTask",
    "RELTask",
    "SentimentTask",
    "SpanCatTask",
    "SummarizationTask",
    "TextCatTask",
]
