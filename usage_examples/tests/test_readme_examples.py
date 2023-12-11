from pathlib import Path
from typing import Callable, Iterable

import pytest
import spacy
from spacy import util
from thinc.compat import has_torch_cuda_gpu

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

        [components.llm.model]
        @llm_models = "spacy.GPT-3-5.v2"
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        doc = nlp("You look gorgeous!")
        print(doc.cats)  # noqa: T201


@pytest.mark.gpu
@pytest.mark.skipif(not has_torch_cuda_gpu, reason="needs GPU & CUDA")
def test_example_2_classifier_hf():
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

        [components.llm.model]
        @llm_models = "spacy.Dolly.v1"
        # For better performance, use databricks/dolly-v2-12b instead
        name = "dolly-v2-3b"
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        doc = nlp("You look gorgeous!")
        print(doc.cats)  # noqa: T201


@pytest.mark.external
def test_example_3_ner():
    examples_path = Path(__file__).parent.parent / "ner_v3_openai" / "examples.json"

    with util.make_tempdir() as tmpdir:

        cfg_str = f"""
        [nlp]
        lang = "en"
        pipeline = ["llm"]

        [components]

        [components.llm]
        factory = "llm"

        [components.llm.task]
        @llm_tasks = "spacy.NER.v3"
        labels = ["DISH", "INGREDIENT", "EQUIPMENT"]
        description = Entities are the names food dishes,
            ingredients, and any kind of cooking equipment.
            Adjectives, verbs, adverbs are not entities.
            Pronouns are not entities.

        [components.llm.task.label_definitions]
        DISH = "Known food dishes, e.g. Lobster Ravioli, garlic bread"
        INGREDIENT = "Individual parts of a food dish, including herbs and spices."
        EQUIPMENT = "Any kind of cooking equipment. e.g. oven, cooking pot, grill"

        [components.llm.task.examples]
        @misc = "spacy.FewShotReader.v1"
        path = {str(examples_path)}

        [components.llm.model]
        @llm_models = "spacy.GPT-3-5.v1"
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        nlp = assemble(tmpdir / "cfg")
        doc = nlp(
            "Sriracha sauce goes really well with hoisin stir fry, "
            "but you should add it after you use the wok."
        )
        print([(ent.text, ent.label_) for ent in doc.ents])  # noqa: T201


@pytest.mark.external
def test_example_4_python():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v3",
                "labels": ["DISH", "INGREDIENT", "EQUIPMENT"],
                "examples": [
                    {
                        "text": "You can't get a great chocolate flavor with carob.",
                        "spans": [
                            {
                                "text": "chocolate",
                                "is_entity": False,
                                "label": "==NONE==",
                                "reason": "is a flavor in this context, not an ingredient",
                            },
                            {
                                "text": "carob",
                                "is_entity": True,
                                "label": "INGREDIENT",
                                "reason": "is an ingredient to add chocolate flavor",
                            },
                        ],
                    },
                ],
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v2",
            },
        },
    )
    nlp.initialize()
    doc = nlp(
        "Sriracha sauce goes really well with hoisin stir fry, "
        "but you should add it after you use the wok."
    )
    print([(ent.text, ent.label_) for ent in doc.ents])  # noqa: T201


def test_example_5_custom_model():
    import random

    @registry.llm_models("RandomClassification.v1")
    def random_textcat(
        labels: str,
    ) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
        labels = labels.split(",")

        def _classify(prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
            for prompts_for_doc in prompts:
                yield [random.choice(labels) for _ in prompts_for_doc]

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

        [components.llm.model]
        @llm_models = "RandomClassification.v1"
        labels = ${components.llm.task.labels}
        """

        with open(tmpdir / "cfg", "w") as text_file:
            text_file.write(cfg_str)

        with pytest.warns(UserWarning, match="Task supports sharding"):
            nlp = assemble(tmpdir / "cfg")
        nlp("i'd like a large margherita pizza please")
