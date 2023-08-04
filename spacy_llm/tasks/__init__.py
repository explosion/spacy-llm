from spacy import Language

from ..pipeline.llm import make_llm
from ..registry import registry
from .lemma import LemmaTask, make_lemma_task
from .ner import NERTask, make_ner_task, make_ner_task_v2
from .noop import NoopTask, make_noop_task
from .rel import RELTask, make_rel_task
from .sentiment import SentimentTask, make_sentiment_task
from .spancat import SpanCatTask, make_spancat_task, make_spancat_task_v2
from .summarization import SummarizationTask, make_summarization_task
from .textcat import TextCatTask, make_textcat_task

# Register llm_TASK factories with NoOp models.
for task_handle in registry.llm_tasks.get_all():
    Language.factory(
        name=f"llm_{task_handle.split('.')[1].lower()}",
        default_config={
            "task": {"@llm_tasks": task_handle},
            "model": {"@llm_models": "spacy.NoOp.v1"},
            "cache": {
                "@llm_misc": "spacy.BatchCache.v1",
                "path": None,
                "batch_size": 64,
                "max_batches_in_mem": 4,
            },
            "save_io": False,
            "validate_types": True,
        },
        func=make_llm,
    )

__all__ = [
    "make_lemma_task",
    "make_ner_task",
    "make_ner_task_v2",
    "make_noop_task",
    "make_rel_task",
    "make_sentiment_task",
    "make_spancat_task",
    "make_spancat_task_v2",
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
