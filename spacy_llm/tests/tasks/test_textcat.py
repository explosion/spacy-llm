# mypy: ignore-errors
import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.registry import lowercase_normalizer, fewshot_reader
from spacy_llm.tasks.textcat import TextCatTask


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks: "spacy.TextCat.v1"
    labels: Recipe
    exclusive_classes: true

    [components.llm.task.normalizer]
    @misc: "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends: "spacy.REST.v1"
    api: "OpenAI"
    config: {}
    """


@pytest.fixture
def fewshot_cfg_string():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks: "spacy.TextCat.v1"
    labels: Recipe
    exclusive_classes: true

    [components.llm.task.examples]
    @misc: "spacy.FewShotReader.v1"
    path: spacy_llm/tests/tasks/examples/textcat_examples.yml

    [components.llm.task.normalizer]
    @misc: "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends: "spacy.REST.v1"
    api: "OpenAI"
    config: {}
    """


@pytest.fixture
def binary():
    text = "Get 1 cup of sugar, half a cup of butter, and mix them together to make a cream"
    labels = "Recipe"
    gold_cats = ["Recipe"]
    exclusive_classes = True
    return text, labels, gold_cats, exclusive_classes


@pytest.fixture
def multilabel():
    text = "You need to increase the temperature when baking, it looks undercooked."
    labels = "Recipe,Feedback,Comment"
    gold_cats = labels.split(",")
    exclusive_classes = True
    return text, labels, gold_cats, exclusive_classes


@pytest.fixture
def multilabel_nonexcl():
    text = "I suggest you add some bananas. Mix 3 pieces of banana to your batter before baking."
    labels = "Recipe,Feedback,Comment"
    gold_cats = labels.split(",")
    exclusive_classes = False
    return text, labels, gold_cats, exclusive_classes


@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_textcat_config(task, cfg_string, request):
    """Simple test to check if the config loads properly given different settings"""
    cfg_string = request.getfixturevalue(cfg_string)
    task = request.getfixturevalue(task)
    _, labels, _, exclusive_classes = task

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
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_textcat_predict(task, cfg_string, request):
    """Use OpenAI to get zero-shot Textcat results
    Note that this test may fail randomly, as the LLM's output is unguaranteed
    to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    task = request.getfixturevalue(task)
    text, labels, gold_cats, exclusive_classes = task
    orig_config = Config().from_str(
        cfg_string,
        overrides={
            "components.llm.task.labels": labels,
            "components.llm.task.exclusive_classes": exclusive_classes,
        },
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    doc = nlp(text)
    assert len(doc.cats) >= 0  # can be 0 if binary and negative
    for cat in list(doc.cats.keys()):
        assert cat in gold_cats


@pytest.mark.external
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_textcat_io(task, cfg_string, request):
    cfg_string = request.getfixturevalue(cfg_string)
    task = request.getfixturevalue(task)
    text, labels, gold_cats, exclusive_classes = task
    orig_config = Config().from_str(
        cfg_string,
        overrides={
            "components.llm.task.labels": labels,
            "components.llm.task.exclusive_classes": exclusive_classes,
        },
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    doc = nlp2(text)
    assert len(doc.cats) >= 0  # can be 0 if binary and negative
    for cat in list(doc.cats.keys()):
        assert cat in gold_cats


def test_textcat_sets_exclusive_classes_if_binary():
    """Test if the textcat task automatically sets exclusive classes to True if binary"""
    llm_textcat = TextCatTask(labels="Recipe", exclusive_classes=False)
    assert llm_textcat._exclusive_classes


@pytest.mark.parametrize(
    "text,response,expected",
    [
        ("Some test text with positive response", "POS", ["Recipe"]),
        ("Some test text with negative response", "NEG", []),
        ("Some test text with weird response", "WeIrD OUtpuT", []),
        ("Some test text with lowercase response", "pos", ["Recipe"]),
        ("Some test text with lowercase response", "neg", []),
    ],
)
def test_textcat_binary_labels_are_correct(text, response, expected):
    """Test if positive label for textcat binary is always the label name and the negative
    label is an empty dictionary
    """
    llm_textcat = TextCatTask(
        labels="Recipe", exclusive_classes=True, normalizer=lowercase_normalizer()
    )

    nlp = spacy.blank("xx")
    doc = nlp(text)
    pred = list(llm_textcat.parse_responses([doc], [response]))[0]
    assert list(pred.cats.keys()) == expected


@pytest.mark.parametrize(
    "text,exclusive_classes,response,expected",
    [
        # fmt: off
        ("Golden path for exclusive", True, "Recipe", ["Recipe"]),
        ("Golden path for non-exclusive", False, "Recipe,Feedback", ["Recipe", "Feedback"]),
        ("Non-exclusive but responded with a single label", False, "Recipe", ["Recipe"]),  # shouldn't matter
        ("Exclusive but responded with multilabel", True, "Recipe,Comment", []),  # don't store anything
        ("Weird outputs for exclusive", True, "reCiPe", ["Recipe"]),
        ("Weird outputs for non-exclusive", False, "reciPE,CoMMeNt,FeedBack", ["Recipe", "Comment", "Feedback"]),
        ("Extra spaces for exclusive", True, "Recipe   ", ["Recipe"]),
        ("Extra spaces for non-exclusive", False, "Recipe,   Comment,    Feedback", ["Recipe", "Comment", "Feedback"]),
        ("One weird value", False, "Recipe,Comment,SomeOtherUnnecessaryLabel", ["Recipe", "Comment"]),
        # fmt: on
    ],
)
def test_textcat_multilabel_labels_are_correct(
    text, exclusive_classes, response, expected
):
    labels = "Recipe,Comment,Feedback"
    llm_textcat = TextCatTask(
        labels=labels,
        exclusive_classes=exclusive_classes,
        normalizer=lowercase_normalizer(),
    )
    nlp = spacy.blank("xx")
    doc = nlp(text)
    pred = list(llm_textcat.parse_responses([doc], [response]))[0]
    # Take only those that have scores
    pred_cats = [cat for cat, score in pred.cats.items() if score == 1.0]
    assert pred_cats == expected


# TODO: test if LLM returns weird results: PoS, NEg, RECIPe
# output: it should still work because we're normalizing


# TODO: test edge cases
# binary, non-exclusive (not sure if it should raise an error) or check if warning happens

# TODO: test potential user errrors
