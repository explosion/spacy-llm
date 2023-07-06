import json
import re
from pathlib import Path

import pytest
import spacy
import srsly
from confection import Config
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, file_reader, lowercase_normalizer
from spacy_llm.registry import strip_normalizer
from spacy_llm.tasks.ner import NERTask, make_ner_task_v2
from spacy_llm.tasks.util import find_substrings
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
    @llm_tasks = "spacy.NER.v1"
    labels = PER,ORG,LOC

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def zeroshot_cfg_string_v2_lds():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v2"
    labels = PER,ORG,LOC

    [components.llm.task.label_definitions]
    PER = "Any named individual in the text"
    ORG = "Any named organization in the text"
    LOC = "The name of any politically or geographically defined location"

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
    @llm_tasks = "spacy.NER.v1"
    labels = PER,ORG,LOC

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.yml"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.fixture
def fewshot_cfg_string_v2():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v2"
    labels = ["PER", "ORG", "LOC"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.yml"))}

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
    @llm_tasks = "spacy.NER.v2"
    labels = PER,ORG,LOC

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "ner.jinja2"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "zeroshot_cfg_string_v2_lds",
        "fewshot_cfg_string",
        "fewshot_cfg_string_v2",
        "ext_template_cfg_string",
    ],
)
def test_ner_config(cfg_string, request):
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]

    # also test nlp config from a dict in add_pipe
    component_cfg = dict(orig_config["components"]["llm"])
    component_cfg.pop("factory")

    nlp2 = spacy.blank("en")
    nlp2.add_pipe("llm", config=component_cfg)
    assert nlp2.pipe_names == ["llm"]

    pipe = nlp.get_pipe("llm")
    assert isinstance(pipe, LLMWrapper)
    assert isinstance(pipe.task, LLMTask)

    labels = orig_config["components"]["llm"]["task"]["labels"]
    labels = split_labels(labels)
    task = pipe.task
    assert isinstance(task, Labeled)
    assert sorted(task.labels) == sorted(tuple(labels))
    assert pipe.labels == task.labels
    assert nlp.pipe_labels["llm"] == list(task.labels)


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "zeroshot_cfg_string_v2_lds",
        "fewshot_cfg_string",
        "fewshot_cfg_string_v2",
        "ext_template_cfg_string",
    ],
)
def test_ner_predict(cfg_string, request):
    """Use OpenAI to get zero-shot NER results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    orig_cfg_string = cfg_string
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)

    if orig_cfg_string != "ext_template_cfg_string":
        assert len(doc.ents) > 0
        for ent in doc.ents:
            assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.external
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "zeroshot_cfg_string_v2_lds",
        "fewshot_cfg_string",
        "fewshot_cfg_string_v2",
        "ext_template_cfg_string",
    ],
)
def test_ner_io(cfg_string, request):
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    text = "Marc and Bob both live in Ireland."
    doc = nlp2(text)
    assert len(doc.ents) >= 0  # can be zero if template is too simple / test-like
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.parametrize(
    "text,input_strings,result_strings,result_offsets",
    [
        (
            "Felipe and Jaime went to the library.",
            ["Felipe", "Jaime", "library"],
            ["Felipe", "Jaime", "library"],
            [(0, 6), (11, 16), (29, 36)],
        ),  # simple
        (
            "The Manila Observatory was founded in 1865 in Manila.",
            ["Manila", "The Manila Observatory"],
            ["Manila", "Manila", "The Manila Observatory"],
            [(4, 10), (46, 52), (0, 22)],
        ),  # overlapping and duplicated
        (
            "Take the road from downtown and turn left at the public market.",
            ["public market", "downtown"],
            ["public market", "downtown"],
            [(49, 62), (19, 27)]
            # flipped
        ),
    ],
)
def test_ensure_offsets_correspond_to_substrings(
    text, input_strings, result_strings, result_offsets
):
    offsets = find_substrings(text, input_strings)
    # Compare strings instead of offsets, but we need to get
    # those strings first from the text
    assert result_offsets == offsets
    found_substrings = [text[start:end] for start, end in offsets]
    assert result_strings == found_substrings


@pytest.mark.parametrize(
    "text,response,gold_ents",
    [
        # simple
        (
            "Jean Jacques and Jaime went to the library.",
            "PER: Jean Jacques, Jaime\nLOC: library",
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "LOC")],
        ),
        # overlapping: should only return the longest span
        (
            "The Manila Observatory was founded in 1865.",
            "LOC: The Manila Observatory, Manila, Manila Observatory",
            [("The Manila Observatory", "LOC")],
        ),
        # flipped: order shouldn't matter
        (
            "Take the road from Downtown and turn left at the public market.",
            "LOC: public market, Downtown",
            [("Downtown", "LOC"), ("public market", "LOC")],
        ),
    ],
)
def test_ner_zero_shot_task(text, response, gold_ents):
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v2(labels=labels)
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list so we get what's inside
    doc_out = list(llm_ner.parse_responses([doc_in], [response]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


@pytest.mark.parametrize(
    "response,normalizer,gold_ents",
    [
        (
            "PER: Jean Jacques, Jaime",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "PER: Jean Jacques, Jaime",
            strip_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "PER: Jean Jacques, Jaime",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "per: Jean Jacques, Jaime",
            strip_normalizer(),
            [],
        ),
        (
            "per: Jean Jacques, Jaime",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "per: Jean Jacques\nPER: Jaime",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "per: Jean Jacques, Jaime\nOrg: library",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "ORG")],
        ),
        (
            "per: Jean Jacques, Jaime\nRANDOM: library",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
    ],
)
def test_ner_labels(response, normalizer, gold_ents):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v2(labels=labels, normalizer=normalizer)
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([doc_in], [response]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


@pytest.mark.parametrize(
    "response,alignment_mode,gold_ents",
    [
        (
            "PER: Jacq",
            "strict",
            [],
        ),
        (
            "PER: Jacq",
            "contract",
            [],
        ),
        (
            "PER: Jacq",
            "expand",
            [("Jacques", "PER")],
        ),
        (
            "PER: Jean J",
            "contract",
            [("Jean", "PER")],
        ),
        (
            "PER: Jean Jacques, aim",
            "strict",
            [("Jean Jacques", "PER")],
        ),
        (
            "PER: random",
            "expand",
            [],
        ),
    ],
)
def test_ner_alignment(response, alignment_mode, gold_ents):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v2(labels=labels, alignment_mode=alignment_mode)
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([doc_in], [response]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


def test_invalid_alignment_mode():
    labels = "PER,ORG,LOC"
    with pytest.raises(ValueError, match="Unsupported alignment mode 'invalid"):
        make_ner_task_v2(labels=labels, alignment_mode="invalid")  # type: ignore


@pytest.mark.parametrize(
    "response,case_sensitive,single_match,gold_ents",
    [
        (
            "PER: Jean",
            False,
            False,
            [("jean", "PER"), ("Jean", "PER"), ("Jean", "PER")],
        ),
        (
            "PER: Jean",
            False,
            True,
            [("jean", "PER")],
        ),
        (
            "PER: Jean",
            True,
            False,
            [("Jean", "PER"), ("Jean", "PER")],
        ),
        (
            "PER: Jean",
            True,
            True,
            [("Jean", "PER")],
        ),
    ],
)
def test_ner_matching(response, case_sensitive, single_match, gold_ents):
    text = "This guy jean (or Jean) is the president of the Jean Foundation."
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v2(
        labels=labels, case_sensitive_matching=case_sensitive, single_match=single_match
    )
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([doc_in], [response]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


def test_jinja_template_rendering_without_examples():
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    llm_ner = make_ner_task_v2(labels=labels, examples=None)
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entities for the set of predefined entity labels.
From the Text input provided, extract named entities for each label in the following format:

LOC: <comma delimited list of strings>
ORG: <comma delimited list of strings>
PER: <comma delimited list of strings>


Here is the text that needs labeling:

Text:
'''
Alice and Bob went to the supermarket
'''
""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "ner.json"),
        str(EXAMPLES_DIR / "ner.yml"),
        str(EXAMPLES_DIR / "ner.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    examples = fewshot_reader(examples_path)
    llm_ner = make_ner_task_v2(labels=labels, examples=examples)
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entities for the set of predefined entity labels.
From the Text input provided, extract named entities for each label in the following format:

LOC: <comma delimited list of strings>
ORG: <comma delimited list of strings>
PER: <comma delimited list of strings>


Below are some examples (only use these as a guide):

Text:
'''
Jack and Jill went up the hill.
'''

PER: Jack, Jill
LOC: hill

Text:
'''
Jack fell down and broke his crown.
'''

PER: Jack

Text:
'''
Jill came tumbling after.
'''

PER: Jill


Here is the text that needs labeling:

Text:
'''
Alice and Bob went to the supermarket
'''""".strip()
    )


def test_jinja_template_rendering_with_label_definitions():
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")
    llm_ner = make_ner_task_v2(
        labels=labels,
        label_definitions={
            "PER": "Person definition",
            "ORG": "Organization definition",
            "LOC": "Location definition",
        },
    )
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entities for the set of predefined entity labels.
From the Text input provided, extract named entities for each label in the following format:

LOC: <comma delimited list of strings>
ORG: <comma delimited list of strings>
PER: <comma delimited list of strings>

Below are definitions of each label to help aid you in what kinds of named entities to extract for each label.
Assume these definitions are written by an expert and follow them closely.

PER: Person definition
ORG: Organization definition
LOC: Location definition


Here is the text that needs labeling:

Text:
'''
Alice and Bob went to the supermarket
'''""".strip()
    )


def test_example_not_following_basemodel():
    wrong_example = [
        {
            "text": "I'm a wrong example. Entities should be a dict, not a list",
            # Should be: {"PER": ["Entities"], "ORG": ["dict", "list"]}
            "entities": [("PER", ("Entities")), ("ORG", ("dict", "list"))],
        }
    ]
    with make_tempdir() as tmpdir:
        tmp_path = tmpdir / "wrong_example.yml"
        srsly.write_yaml(tmp_path, wrong_example)

        with pytest.raises(ValueError):
            make_ner_task_v2(labels="PER,ORG,LOC", examples=fewshot_reader(tmp_path))


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "ner.jinja2")
    template = file_reader(template_path)
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    llm_ner = make_ner_task_v2(labels=labels, template=template)
    prompt = list(llm_ner.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
This is a test NER template. Here are the labels
LOC
ORG
PER

Here is the text: Alice and Bob went to the supermarket
""".strip()
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
    @llm_tasks = "spacy.NER.v2"
    labels = PER,ORG,LOC

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    output = "PER: Alice,Bob"
    """


@pytest.mark.parametrize("n_detections", [0, 1, 2])
def test_ner_scoring(noop_config, n_detections):
    config = Config().from_str(noop_config)
    nlp = assemble_from_config(config)

    examples = []

    for text in ["Alice works with Bob.", "Bob lives with Alice."]:
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        reference.ents = [
            Span(reference, 0, 1, label="PER"),
            Span(reference, 3, 4, label="PER"),
        ][:n_detections]

        examples.append(Example(predicted, reference))

    scores = nlp.evaluate(examples)

    assert scores["ents_p"] == n_detections / 2


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_ner_init(noop_config, n_prompt_examples: int):
    config = Config().from_str(noop_config)
    del config["components"]["llm"]["task"]["labels"]

    nlp = assemble_from_config(config)

    examples = []

    for text in [
        "Alice works with Bob in London.",
        "Bob lives with Alice in Manchester.",
    ]:
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        reference.ents = [
            Span(reference, 0, 1, label="PER"),
            Span(reference, 3, 4, label="PER"),
            Span(reference, 5, 6, label="LOC"),
        ]

        examples.append(Example(predicted, reference))

    _, llm = nlp.pipeline[0]
    task: NERTask = llm._task

    assert set(task._label_dict.values()) == set()
    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }
    nlp.initialize(lambda: examples)

    assert set(task._label_dict.values()) == {"PER", "LOC"}
    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(examples)

    if n_prompt_examples > 0:
        for eg in task._prompt_examples:
            assert set(eg.entities.keys()) == {"PER", "LOC"}


def test_ner_serde(noop_config):
    config = Config().from_str(noop_config)
    del config["components"]["llm"]["task"]["labels"]

    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "per": "PER"}

    task1: NERTask = nlp1.get_pipe("llm")._task
    task2: NERTask = nlp2.get_pipe("llm")._task

    # Artificially add labels to task1
    task1._label_dict = labels

    assert task1._label_dict == labels
    assert task2._label_dict == dict()

    b = nlp1.to_bytes()
    nlp2.from_bytes(b)

    assert task1._label_dict == task2._label_dict == labels


def test_ner_to_disk(noop_config, tmp_path: Path):
    config = Config().from_str(noop_config)
    del config["components"]["llm"]["task"]["labels"]

    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "per": "PER"}

    task1: NERTask = nlp1.get_pipe("llm")._task
    task2: NERTask = nlp2.get_pipe("llm")._task

    # Artificially add labels to task1
    task1._label_dict = labels

    assert task1._label_dict == labels
    assert task2._label_dict == dict()

    path = tmp_path / "model"

    nlp1.to_disk(path)

    cfgs = list(path.rglob("cfg"))
    assert len(cfgs) == 1

    cfg = json.loads(cfgs[0].read_text())
    assert cfg["_label_dict"] == labels

    nlp2.from_disk(path)

    assert task1._label_dict == task2._label_dict == labels


def test_label_inconsistency():
    """Test whether inconsistency between specified labels and labels in examples is detected."""
    cfg = f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v2"
    labels = ["PERSON", "LOCATION"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner_inconsistent.yml"))}

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    """

    config = Config().from_str(cfg)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Examples contain labels that are not specified in the task configuration. The latter contains the "
            "following labels: ['LOCATION', 'PERSON']. Labels in examples missing from the task configuration: "
            "['TECH']. Please ensure your label specification and example labels are consistent."
        ),
    ):
        nlp = assemble_from_config(config)

    prompt_examples = nlp.get_pipe("llm")._task._prompt_examples
    assert len(prompt_examples) == 2
    assert prompt_examples[0].text == "Jack and Jill went up the hill."
    assert prompt_examples[0].entities == {
        "LOCATION": ["hill"],
        "PERSON": ["Jack", "Jill"],
    }
    assert (
        prompt_examples[1].text
        == "Jack and Jill went up the hill and spaCy is a great tool."
    )
    assert prompt_examples[1].entities == {
        "LOCATION": ["hill"],
        "PERSON": ["Jack", "Jill"],
    }
