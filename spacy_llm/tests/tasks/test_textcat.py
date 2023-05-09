import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.tasks.textcat import textcat_zeroshot_task
from spacy_llm.registry import noop_normalizer, lowercase_normalizer


cfg_string = """
[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks: "spacy.TextCatZeroShot.v1"
labels: Recipe
exclusive_classes: true

[components.llm.task.normalizer]
@misc: "spacy.LowercaseNormalizer.v1"

[components.llm.backend]
@llm_backends: "spacy.MiniChain.v1"
api: "OpenAI"
config: {}
"""


@pytest.mark.parametrize(
    "labels,exclusive_classes",
    [
        ("Recipe", True),
        ("Recipe,Feedback,Comment", True),
        ("Recipe,Feedback,Comment", False),
    ],
)
def test_textcat_config(labels: str, exclusive_classes: bool):
    """Simple test to check if the config loads properly given different settings"""
    orig_config = Config().from_str(
        cfg_string,
        overrides={
            "components.llm.task.labels": labels,
            "components.llm.task.exclusive_classes": exclusive_classes,
        },
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


@pytest.mark.external
@pytest.mark.parametrize(
    "text,labels,exclusive_classes,gold_cats",
    [
        (
            "Get 1 cup of sugar, half a cup of butter, and mix them together to make a cream",
            "Recipe",
            True,
            ["POS", "NEG"],
        ),
        (
            "You might need to increase the temperature when baking, it looks undercooked.",
            "Recipe,Feedback,Comment",
            True,
            ["Recipe", "Feedback", "Comment"],
        ),
        (
            "I suggest you add some bananas. Mix 3 pieces of banana to your batter before baking.",
            "Recipe,Feedback,Comment",
            False,
            ["Recipe", "Feedback", "Comment"],
        ),
    ],
)
def test_textcat_predict(text, labels, exclusive_classes, gold_cats):
    """Use OpenAI to get zero-shot Textcat results
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    orig_config = Config().from_str(
        cfg_string,
        overrides={
            "components.llm.task.labels": labels,
            "components.llm.task.exclusive_classes": exclusive_classes,
        },
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    doc = nlp(text)
    assert len(doc.cats) > 0
    assert set(doc.cats.keys()) == set(gold_cats)


# TODO: Test golden paths
# binary, exclusive
# multilabel exclusive
# multilabel non-exclusive

# TODO: test if LLM returns weird results: PoS, NEg, RECIPe
# output: it should still work because we're normalizing

# TODO: test if LLM returns wild results: POSITIVE, NEGATIVE
# should fail gracefully

# TODO: test edge cases
# binary, non-exclusive (not sure if it should raise an error) or check if warning happens

# TODO: test potential user errrors


# TODO: test IO

# TODO: test external with OpenAI
