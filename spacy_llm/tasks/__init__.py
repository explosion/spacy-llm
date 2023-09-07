from spacy import Language

from ..pipeline.llm import DEFAULT_CACHE_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_SAVE_IO
from ..pipeline.llm import DEFAULT_VALIDATE_TYPES, make_llm
from .builtin_task import BuiltinTask
from .lemma import LemmaTask, make_lemma_task
from .ner import NERTask, make_ner_task_v3
from .noop import NoopTask, make_noop_task
from .rel import RELTask, make_rel_task
from .sentiment import SentimentTask, make_sentiment_task
from .spancat import SpanCatTask, make_spancat_task_v3
from .summarization import SummarizationTask, make_summarization_task
from .textcat import TextCatTask, make_textcat_task

_LATEST_TASKS = (
    "spacy.NER.v3",
    "spacy.REL.v1",
    "spacy.Sentiment.v1",
    "spacy.SpanCat.v3",
    "spacy.Summarization.v1",
    "spacy.TextCat.v3",
)

# Register llm_TASK factories with default models.
for task_handle in _LATEST_TASKS:
    Language.factory(
        name=f"llm_{task_handle.split('.')[1].lower()}",
        default_config={
            "task": {"@llm_tasks": task_handle},
            "model": DEFAULT_MODEL_CONFIG,
            "cache": DEFAULT_CACHE_CONFIG,
            "save_io": DEFAULT_SAVE_IO,
            "validate_types": DEFAULT_VALIDATE_TYPES,
        },
        func=make_llm,
    )

__all__ = [
    "make_lemma_task",
    "make_ner_task_v3",
    "make_noop_task",
    "make_rel_task",
    "make_sentiment_task",
    "make_spancat_task_v3",
    "make_summarization_task",
    "make_textcat_task",
    "BuiltinTask",
    "LemmaTask",
    "NERTask",
    "NoopTask",
    "RELTask",
    "SentimentTask",
    "SpanCatTask",
    "SummarizationTask",
    "TextCatTask",
]
