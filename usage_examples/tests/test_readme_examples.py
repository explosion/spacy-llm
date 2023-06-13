from typing import Iterable, Callable

import pytest
import spacy
from thinc.compat import has_torch_cuda_gpu

from spacy import util
from spacy_llm.registry import registry
from spacy_llm.util import assemble


@pytest.mark.external
def test_example_1_classifier():
    with util.make_tempdir() as tmpdir:
        cfg_str = """
        [nlp]
        lang = "en"
        pipeline = ["llm"]

        [components]

        [components.llm]
        factory = "llm"

        [components.llm.task]
        @llm_tasks = "spacy.TextCat.v2"
        labels = ["COMPLIMENT", "INSULT"]

        [components.llm.backend]
        @llm_backends = "spacy.REST.v1"
        api = "OpenAI"
        config = {"model": "gpt-3.5-turbo", "temperature": 0.3}
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        doc = nlp("You look gorgeous!")
        print(doc.cats)  # noqa: T201


@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_example_2_ner_hf():
    with util.make_tempdir() as tmpdir:
        cfg_str = """
        [nlp]
        lang = "en"
        pipeline = ["llm"]

        [components]

        [components.llm]
        factory = "llm"

        [components.llm.task]
        @llm_tasks = "spacy.NER.v2"
        labels = ["PERSON", "ORGANISATION", "LOCATION"]

        [components.llm.backend]
        @llm_backends = "spacy.Dolly_HF.v1"
        # For better performance, use databricks/dolly-v2-12b instead
        model = "databricks/dolly-v2-3b"
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
        print([(ent.text, ent.label_) for ent in doc.ents])  # noqa: T201


@pytest.mark.external
def test_example_3_python():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v2",
                "labels": ["PERSON", "ORGANISATION", "LOCATION"],
            },
            "backend": {
                "@llm_backends": "spacy.REST.v1",
                "api": "OpenAI",
                "config": {"model": "gpt-3.5-turbo"},
            },
        },
    )
    nlp.initialize()
    doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
    print([(ent.text, ent.label_) for ent in doc.ents])  # noqa: T201


def test_example_4_custom_backend():
    import random

    @registry.llm_backends("RandomClassification.v1")
    def random_textcat(labels: str) -> Callable[[Iterable[str]], Iterable[str]]:
        labels = labels.split(",")

        def _classify(prompts: Iterable[str]) -> Iterable[str]:
            for _ in prompts:
                yield random.choice(labels)

        return _classify

    with util.make_tempdir() as tmpdir:
        cfg_str = """
        [nlp]
        lang = "en"
        pipeline = ["llm"]

        [components]

        [components.llm]
        factory = "llm"

        [components.llm.task]
        @llm_tasks = "spacy.TextCat.v2"
        labels = ORDER,INFORMATION

        [components.llm.backend]
        @llm_backends = "RandomClassification.v1"
        labels = ${components.llm.task.labels}
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        nlp("i'd like a large margherita pizza please")
