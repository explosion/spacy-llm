# mypy: ignore-errors
from pathlib import Path

import pytest
import spacy
import srsly
from confection import Config
from pydantic import ValidationError
from spacy.util import make_tempdir

from spacy_llm.registry import lowercase_normalizer, fewshot_reader
from spacy_llm.tasks.textcat import TextCatTask
from spacy_llm.compat import has_openai_key


EXAMPLES_DIR = Path(__file__).parent / "examples"


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
    @llm_tasks = "spacy.TextCat.v1"
    labels = "Recipe"
    exclusive_classes = true

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"
    config = {}
    """


@pytest.fixture
def fewshot_cfg_string():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.TextCat.v1"
    labels = "Recipe"
    exclusive_classes = true

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str(EXAMPLES_DIR / "textcat_examples.yml")}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"
    config = {{}}
    """


@pytest.fixture
def binary():
    text = "Get 1 cup of sugar, half a cup of butter, and mix them together to make a cream"
    labels = "Recipe"
    gold_cats = ["Recipe"]
    exclusive_classes = True
    examples_path = str(EXAMPLES_DIR / "textcat_binary_examples.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.fixture
def multilabel_excl():
    text = "You need to increase the temperature when baking, it looks undercooked."
    labels = "Recipe,Feedback,Comment"
    gold_cats = ["Recipe", "Feedback", "Comment"]
    exclusive_classes = True
    examples_path = str(EXAMPLES_DIR / "textcat_multi_excl_examples.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.fixture
def multilabel_nonexcl():
    text = "I suggest you add some bananas. Mix 3 pieces of banana to your batter before baking."
    labels = "Recipe,Feedback,Comment"
    gold_cats = ["Recipe", "Feedback", "Comment"]
    exclusive_classes = False
    examples_path = str(EXAMPLES_DIR / "textcat_multi_nonexcl_examples.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_textcat_config(task, cfg_string, request):
    """Simple test to check if the config loads properly given different settings"""

    task = request.getfixturevalue(task)
    _, labels, _, exclusive_classes, examples = task
    overrides = {
        "components.llm.task.labels": labels,
        "components.llm.task.exclusive_classes": exclusive_classes,
    }

    if cfg_string == "fewshot_cfg_string":
        overrides["components.llm.task.examples.path"] = examples

    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string, overrides=overrides)
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
    task = request.getfixturevalue(task)
    text, labels, gold_cats, exclusive_classes, examples = task
    overrides = {
        "components.llm.task.labels": labels,
        "components.llm.task.exclusive_classes": exclusive_classes,
    }

    if cfg_string == "fewshot_cfg_string":
        overrides["components.llm.task.examples.path"] = examples

    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string, overrides=overrides)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    doc = nlp(text)
    assert len(doc.cats) >= 0  # can be 0 if binary and negative
    for cat in list(doc.cats.keys()):
        assert cat in gold_cats


@pytest.mark.external
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_textcat_io(task, cfg_string, request):
    task = request.getfixturevalue(task)
    text, labels, gold_cats, exclusive_classes, examples = task
    overrides = {
        "components.llm.task.labels": labels,
        "components.llm.task.exclusive_classes": exclusive_classes,
    }

    if cfg_string == "fewshot_cfg_string":
        overrides["components.llm.task.examples.path"] = examples

    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string, overrides=overrides)

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
    "text,response,expected_score",
    [
        ("Some test text with positive response", "POS", 1.0),
        ("Some test text with negative response", "NEG", 0.0),
        ("Some test text with weird response", "WeIrD OUtpuT", 0.0),
        ("Some test text with lowercase response", "pos", 1.0),
        ("Some test text with lowercase response", "neg", 0.0),
    ],
)
def test_textcat_binary_labels_are_correct(text, response, expected_score):
    """Test if positive label for textcat binary is always the label name and the negative
    label is an empty dictionary
    """
    label = "Recipe"
    llm_textcat = TextCatTask(
        labels=label, exclusive_classes=True, normalizer=lowercase_normalizer()
    )

    nlp = spacy.blank("xx")
    doc = nlp(text)
    pred = list(llm_textcat.parse_responses([doc], [response]))[0]
    assert list(pred.cats.keys())[0] == label
    assert list(pred.cats.values())[0] == expected_score


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
    doc = nlp.make_doc(text)
    pred = list(llm_textcat.parse_responses([doc], [response]))[0]
    # Take only those that have scores
    pred_cats = [cat for cat, score in pred.cats.items() if score == 1.0]
    assert pred_cats == expected


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "textcat_binary_examples.json"),
        str(EXAMPLES_DIR / "textcat_binary_examples.yml"),
        str(EXAMPLES_DIR / "textcat_binary_examples.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples_for_binary(examples_path, binary):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    text, labels, _, exclusive_classes, _ = binary
    nlp = spacy.blank("xx")
    doc = nlp(text)

    examples = fewshot_reader(examples_path)
    llm_textcat = TextCatTask(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
Classify whether the text below belongs to the Recipe category or not.
If it is a Recipe, answer `POS`. If it is not a Recipe, answer `NEG`.
Below are some examples (only use these as a guide):


Text:
'''
Macaroni and cheese is the best budget meal for students, unhealthy tho
'''

NEG

Text:
'''
2 cups soy sauce, 1/2 lb. of chicken, 1/2 cup vinegar, then salt and paper, mix then well and you get an adobo
'''

POS

Text:
'''
You can still add more layers to that croissant, get extra butter and add a few cups of flour
'''

POS


Here is the text that needs classification


Text:
'''
Get 1 cup of sugar, half a cup of butter, and mix them together to make a cream
'''""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "textcat_multi_excl_examples.json"),
        str(EXAMPLES_DIR / "textcat_multi_excl_examples.yml"),
        str(EXAMPLES_DIR / "textcat_multi_excl_examples.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples_for_multilabel_exclusive(
    examples_path, multilabel_excl
):
    text, labels, _, exclusive_classes, _ = multilabel_excl
    nlp = spacy.blank("xx")
    doc = nlp(text)

    examples = fewshot_reader(examples_path)
    llm_textcat = TextCatTask(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
Classify the text below to any of the following labels: Recipe, Feedback, Comment
The task is exclusive, so only choose one label from what I provided.
Below are some examples (only use these as a guide):


Text:
'''
Macaroni and cheese is the best budget meal for students, unhealthy tho
'''

Comment

Text:
'''
2 cups soy sauce, 1/2 lb. of chicken, 1/2 cup vinegar, then salt and paper, mix then well and you get an adobo
'''

Recipe

Text:
'''
You can still add more layers to that croissant, get extra butter and add a few cups of flour
'''

Feedback


Here is the text that needs classification


Text:
'''
You need to increase the temperature when baking, it looks undercooked.
'''""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "textcat_multi_nonexcl_examples.json"),
        str(EXAMPLES_DIR / "textcat_multi_nonexcl_examples.yml"),
        str(EXAMPLES_DIR / "textcat_multi_nonexcl_examples.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples_for_multilabel_nonexclusive(
    examples_path, multilabel_nonexcl
):
    text, labels, _, exclusive_classes, _ = multilabel_nonexcl
    nlp = spacy.blank("xx")
    doc = nlp(text)

    examples = fewshot_reader(examples_path)
    llm_textcat = TextCatTask(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
    Classify the text below to any of the following labels: Recipe, Feedback, Comment
The task is non-exclusive, so you can provide more than one label as long as
they're comma-delimited. For example: Label1, Label2, Label3.
If the text cannot be classified into any of the provided labels, answer `==NONE==`.
Below are some examples (only use these as a guide):


Text:
'''
Macaroni and cheese is the best budget meal for students, unhealthy tho
'''

Comment,Feedback

Text:
'''
2 cups soy sauce, 1/2 lb. of chicken, 1/2 cup vinegar, then salt and paper, mix then well and you get an adobo
'''

Recipe

Text:
'''
You can still add more layers to that croissant, get extra butter and add a few cups of flour
'''

Feedback,Recipe


Here is the text that needs classification


Text:
'''
I suggest you add some bananas. Mix 3 pieces of banana to your batter before baking.
'''""".strip()
    )


@pytest.mark.parametrize(
    "wrong_example,labels,exclusive_classes",
    [
        # fmt: off
        ([{"text": "wrong example for binary", "answer": [0]}], "Label", True),
        ([{"text": "wrong example for multilabel excl", "answer": [12345]}], "Label1,Label2", True),
        ([{"text": "wrong example for multilabel nonexcl", "answer": ["Label1", "Label2"]}], "Label,Label2", False),
        # fmt: on
    ],
)
def test_example_not_following_basemodel(wrong_example, labels, exclusive_classes):
    with make_tempdir() as tmpdir:
        tmp_path = tmpdir / "wrong_example.yml"
        srsly.write_yaml(tmp_path, wrong_example)

        with pytest.raises(ValidationError):
            TextCatTask(
                labels=labels,
                examples=fewshot_reader(tmp_path),
                exclusive_classes=exclusive_classes,
            )
