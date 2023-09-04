from spacy import Language

from ..pipeline.llm import make_llm
from .builtin_task import BuiltinTask
from .entity_linker import EntityLinkerTask, make_entitylinker_task
from .lemma import LemmaTask, make_lemma_task
from .ner import NERTask, make_ner_task_v3
from .noop import NoopTask, make_noop_task
from .rel import RELTask, make_rel_task
from .sentiment import SentimentTask, make_sentiment_task
from .spancat import SpanCatTask, make_spancat_task_v3
from .summarization import SummarizationTask, make_summarization_task
from .textcat import TextCatTask, make_textcat_task

_LATEST_TASKS = (
    "spacy.EntityLinker.v1",
    "spacy.NER.v3",
    "spacy.REL.v1",
    "spacy.Sentiment.v1",
    "spacy.SpanCat.v3",
    "spacy.Summarization.v1",
    "spacy.TextCat.v3",
)

# Register llm_TASK factories with NoOp models.
for task_handle in _LATEST_TASKS:
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
    "make_entitylinker_task",
    "make_lemma_task",
    "make_ner_task_v3",
    "make_noop_task",
    "make_rel_task",
    "make_sentiment_task",
    "make_spancat_task_v3",
    "make_summarization_task",
    "make_textcat_task",
    "BuiltinTask",
    "EntityLinkerTask",
    "LemmaTask",
    "NERTask",
    "NoopTask",
    "RELTask",
    "SentimentTask",
    "SpanCatTask",
    "SummarizationTask",
    "TextCatTask",
]
