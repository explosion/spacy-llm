import json
from pathlib import Path
from typing import Iterable

import pytest
import spacy
import srsly
from confection import Config
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, file_reader, lowercase_normalizer
from spacy_llm.registry import registry
from spacy_llm.tasks.textcat import TextCatTask, make_textcat_task_v3
from spacy_llm.ty import Labeled, LLMTask
from spacy_llm.util import assemble_from_config, split_labels

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


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

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
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
    path = {str(EXAMPLES_DIR / "textcat.yml")}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def ext_template_cfg_string():
    """Simple zero-shot config with an external template"""

    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.TextCat.v2"
    labels = ["Recipe"]
    exclusive_classes = true

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "textcat.jinja2"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def zeroshot_cfg_string_v3_lds():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.TextCat.v3"
    labels = "Recipe"
    exclusive_classes = true

    [components.llm.task.label_definitions]
    Recipe = "A recipe is a set of instructions for preparing a meal, including a list of the ingredients required."

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def binary():
    text = "Get 1 cup of sugar, half a cup of butter, and mix them together to make a cream"
    labels = "Recipe"
    gold_cats = ["Recipe"]
    exclusive_classes = True
    examples_path = str(EXAMPLES_DIR / "textcat_binary.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.fixture
def multilabel_excl():
    text = "You need to increase the temperature when baking, it looks undercooked."
    labels = "Recipe,Feedback,Comment"
    gold_cats = ["Recipe", "Feedback", "Comment"]
    exclusive_classes = True
    examples_path = str(EXAMPLES_DIR / "textcat_multi_excl.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.fixture
def multilabel_nonexcl():
    text = "I suggest you add some bananas. Mix 3 pieces of banana to your batter before baking."
    labels = "Recipe,Feedback,Comment"
    gold_cats = ["Recipe", "Feedback", "Comment"]
    exclusive_classes = False
    examples_path = str(EXAMPLES_DIR / "textcat_multi_nonexcl.yml")
    return text, labels, gold_cats, exclusive_classes, examples_path


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "task",
    ["binary", "multilabel_nonexcl", "multilabel_excl"],
)
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
        "ext_template_cfg_string",
        "zeroshot_cfg_string_v3_lds",
    ],
)
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

    pipe = nlp.get_pipe("llm")
    assert isinstance(pipe, LLMWrapper)
    assert isinstance(pipe.task, LLMTask)

    labels = split_labels(labels)
    task = pipe.task
    assert isinstance(task, Labeled)
    assert sorted(task.labels) == sorted(tuple(labels))
    assert pipe.labels == task.labels
    assert nlp.pipe_labels["llm"] == list(task.labels)


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
        "ext_template_cfg_string",
        "zeroshot_cfg_string_v3_lds",
    ],
)
def test_textcat_predict(task, cfg_string, request):
    """Use OpenAI to get Textcat results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed
    to be consistent/predictable.
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
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("task", ["binary", "multilabel_nonexcl", "multilabel_excl"])
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
        "ext_template_cfg_string",
        "zeroshot_cfg_string_v3_lds",
    ],
)
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
    llm_textcat = make_textcat_task_v3(labels="Recipe", exclusive_classes=False)
    assert llm_textcat._exclusive_classes


@pytest.mark.parametrize(
    "text,response,expected_score",
    [
        ("Some test text with positive response", "POS", 1.0),
        ("Some test text with negative response", "NEG", 0.0),
        ("Some test text with weird response", "WeIrD OUtpuT", 0.0),
        ("Some test text with lowercase response", "pos", 1.0),
        ("Some test text with lowercase response", "neg", 0.0),
        ("Some test text with unstripped response", "\n\n\nPOS", 1.0),
        ("Some test text with unstripped response", "\n\n\nNEG", 0.0),
    ],
)
def test_textcat_binary_labels_are_correct(text, response, expected_score):
    """Test if positive label for textcat binary is always the label name and the negative
    label is an empty dictionary
    """
    label = "Recipe"
    llm_textcat = make_textcat_task_v3(
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
    llm_textcat = make_textcat_task_v3(
        labels=labels,
        exclusive_classes=exclusive_classes,
        normalizer=lowercase_normalizer(),
    )
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(text)
    pred = list(llm_textcat.parse_responses([doc], [response]))[0]
    # Take only those that have scores
    pred_cats = [cat for cat, score in pred.cats.items() if score == 1.0]
    assert set(pred_cats) == set(expected)


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "textcat_binary.json"),
        str(EXAMPLES_DIR / "textcat_binary.yml"),
        str(EXAMPLES_DIR / "textcat_binary.jsonl"),
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
    llm_textcat = make_textcat_task_v3(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
You are an expert Text Classification system. Your task is to accept Text as input
and provide a category for the text based on the predefined labels.

Classify whether the text below belongs to the Recipe category or not.
If it is a Recipe, answer `POS`. If it is not a Recipe, answer `NEG`.
Do not put any other text in your answer, only one of 'POS' or 'NEG' with nothing before or after.
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
        str(EXAMPLES_DIR / "textcat_multi_excl.json"),
        str(EXAMPLES_DIR / "textcat_multi_excl.yml"),
        str(EXAMPLES_DIR / "textcat_multi_excl.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples_for_multilabel_exclusive(
    examples_path, multilabel_excl
):
    text, labels, _, exclusive_classes, _ = multilabel_excl
    nlp = spacy.blank("xx")
    doc = nlp(text)

    examples = fewshot_reader(examples_path)
    llm_textcat = make_textcat_task_v3(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
You are an expert Text Classification system. Your task is to accept Text as input
and provide a category for the text based on the predefined labels.

Classify the text below to any of the following labels: Comment, Feedback, Recipe

The task is exclusive, so only choose one label from what I provided.
Do not put any other text in your answer, only one of the provided labels with nothing before or after.
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
        str(EXAMPLES_DIR / "textcat_multi_nonexcl.json"),
        str(EXAMPLES_DIR / "textcat_multi_nonexcl.yml"),
        str(EXAMPLES_DIR / "textcat_multi_nonexcl.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples_for_multilabel_nonexclusive(
    examples_path, multilabel_nonexcl
):
    text, labels, _, exclusive_classes, _ = multilabel_nonexcl
    nlp = spacy.blank("xx")
    doc = nlp(text)

    examples = fewshot_reader(examples_path)
    llm_textcat = make_textcat_task_v3(
        labels=labels,
        examples=examples,
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
You are an expert Text Classification system. Your task is to accept Text as input
and provide a category for the text based on the predefined labels.

Classify the text below to any of the following labels: Comment, Feedback, Recipe

The task is non-exclusive, so you can provide more than one label as long as
they're comma-delimited. For example: Label1, Label2, Label3.
Do not put any other text in your answer, only one or more of the provided labels with nothing before or after.
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

        with pytest.raises(ValueError):
            make_textcat_task_v3(
                labels=labels,
                examples=fewshot_reader(tmp_path),
                exclusive_classes=exclusive_classes,
            )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "textcat.jinja2")
    template = file_reader(template_path)
    labels = "Recipe"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Combine 2 cloves of garlic with soy sauce")

    llm_textcat = make_textcat_task_v3(labels=labels, template=template)
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
This is a test textcat template. Here is/are the label/s
Recipe

Here is the text: Combine 2 cloves of garlic with soy sauce
""".strip()
    )


INSULTS = [
    "Gobbledygooks!",
    "Filibusters!",
    "Slubberdegullions!",
    "Vampires!",
    "Sycophant!",
    "Kleptomaniacs!",
    "Egoists!",
    "Tramps!",
    "Monopolizers!",
    "Pockmarks!",
    "Belemnite!",
    "Crooks!",
    "Miserable earthworms!",
    "Harlequin!",
    "Parasites!",
    "Macrocephalic baboon!",
    "Brutes!",
    "Pachyrhizus!",
    "Toads!",
    "Gyroscope!",
    "Bougainvillea!",
    "Bloodsuckers!",
    "Nincompoop!",
    "Shipwreckers!",
]


@pytest.mark.parametrize("n_insults", range(len(INSULTS) + 1))
def test_textcat_scoring(zeroshot_cfg_string, n_insults):
    @registry.llm_models("Dummy")
    def factory():
        def b(prompts: Iterable[str]) -> Iterable[str]:
            for _ in prompts:
                yield "POS"

        return b

    config = Config().from_str(zeroshot_cfg_string)
    config["components"]["llm"]["model"] = {"@llm_models": "Dummy"}
    config["components"]["llm"]["task"]["labels"] = "Insult"
    nlp = assemble_from_config(config)

    examples = []

    for i, text in enumerate(INSULTS):
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        if i < n_insults:
            reference.cats = {"Insult": 1.0}

        examples.append(Example(predicted, reference))

    scores = nlp.evaluate(examples)

    pos = n_insults / len(INSULTS)

    assert scores["cats_micro_p"] == pos
    assert not n_insults or scores["cats_micro_r"] == 1


def test_jinja_template_rendering_with_label_definitions(multilabel_excl):
    text, labels, _, exclusive_classes, _ = multilabel_excl
    nlp = spacy.blank("xx")
    doc = nlp(text)

    llm_textcat = make_textcat_task_v3(
        labels=labels,
        label_definitions={
            "Recipe": "A Recipe is a set of instructions to make a food of some kind",
            "Feedback": "Feedback description",
            "Comment": "Comment description",
        },
        exclusive_classes=exclusive_classes,
    )
    prompt = list(llm_textcat.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
You are an expert Text Classification system. Your task is to accept Text as input
and provide a category for the text based on the predefined labels.

Classify the text below to any of the following labels: Comment, Feedback, Recipe

The task is exclusive, so only choose one label from what I provided.
Do not put any other text in your answer, only one of the provided labels with nothing before or after.

Below are definitions of each label to help aid you in correctly classifying the text.
Assume these definitions are written by an expert and follow them closely.

Recipe: A Recipe is a set of instructions to make a food of some kind
Feedback: Feedback description
Comment: Comment description


Here is the text that needs classification


Text:
'''
You need to increase the temperature when baking, it looks undercooked.
'''""".strip()
    )


@pytest.fixture
def noop_config():
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

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    """


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
@pytest.mark.parametrize("init_from_config", [True, False])
def test_textcat_init(
    noop_config,
    init_from_config: bool,
    n_prompt_examples: bool,
):
    config = Config().from_str(noop_config)
    if init_from_config:
        config["initialize"] = {"components": {"llm": {"labels": ["Test"]}}}
    nlp = assemble_from_config(config)

    examples = []

    for i, text in enumerate(INSULTS):
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        if i < (len(INSULTS) // 2):
            reference.cats = {"Insult": 1.0, "Compliment": 0.0}
        else:
            reference.cats = {"Insult": 0.0, "Compliment": 1.0}

        examples.append(Example(predicted, reference))

    _, llm = nlp.pipeline[0]
    task: TextCatTask = llm._task

    if init_from_config:
        target = {"Test"}
    else:
        target = set()
    assert set(task._label_dict.values()) == target
    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }

    nlp.initialize(lambda: examples)

    if init_from_config:
        target = {"Test"}
    else:
        target = {"Insult", "Compliment"}
    assert set(task._label_dict.values()) == target
    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(INSULTS)


def test_textcat_serde(noop_config, tmp_path: Path):
    config = Config().from_str(noop_config)

    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)
    nlp3 = assemble_from_config(config)

    labels = {"insult": "INSULT", "compliment": "COMPLIMENT"}

    task1: TextCatTask = nlp1.get_pipe("llm")._task
    task2: TextCatTask = nlp2.get_pipe("llm")._task
    task3: TextCatTask = nlp3.get_pipe("llm")._task

    # Artificially add labels to task1
    task1._label_dict = labels

    assert task1._label_dict == labels
    assert task2._label_dict == dict()
    assert task3._label_dict == dict()

    path = tmp_path / "model"

    nlp1.to_disk(path)

    cfgs = list(path.rglob("cfg"))
    assert len(cfgs) == 1

    cfg = json.loads(cfgs[0].read_text())
    assert cfg["_label_dict"] == labels

    nlp2.from_disk(path)
    nlp3.from_bytes(nlp1.to_bytes())

    assert task1._label_dict == task2._label_dict == task3._label_dict == labels
