import json
import re
from pathlib import Path
from typing import Callable, List, Tuple, cast

import pytest
import spacy
import srsly
from confection import Config
from spacy.language import Language
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.compat import Literal, ValidationError
from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, file_reader, lowercase_normalizer
from spacy_llm.registry import strip_normalizer
from spacy_llm.tasks.ner import NERTask, make_ner_task_v3
from spacy_llm.tasks.span import SpanReason
from spacy_llm.tasks.span.parser import _extract_span_reasons_cot
from spacy_llm.tasks.util import find_substrings
from spacy_llm.ty import LabeledTask, ShardingLLMTask
from spacy_llm.util import assemble_from_config, split_labels

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@pytest.fixture
def examples_dir():
    return EXAMPLES_DIR


@pytest.fixture
def templates_dir():
    return TEMPLATES_DIR


@pytest.fixture
def noop_config():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v3"
    labels = PER,ORG,LOC

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.json"))}

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    output = 1. Bob | True | PER | is the name of a person
        2. Alice | True | PER | is the name of a person
    """


@pytest.fixture
def fewshot_cfg_string_v3_lds():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v3"
    description = "This is a description"
    labels = PER,ORG,LOC

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.json"))}

    [components.llm.task.label_definitions]
    PER = "Any named individual in the text"
    ORG = "Any named organization in the text"
    LOC = "The name of any politically or geographically defined location"

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v2"
    """


@pytest.fixture
def fewshot_cfg_string_v3():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.NER.v3"
    description = "This is a description"
    labels = ["PER", "ORG", "LOC"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.json"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v2"
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
    @llm_tasks = "spacy.NER.v3"
    description = "This is a description"
    labels = ["PER", "ORG", "LOC"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner.json"))}

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "ner.jinja2"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v2"
    """


@pytest.fixture(
    params=[
        "fewshot_cfg_string_v3_lds",
        "fewshot_cfg_string_v3",
        "ext_template_cfg_string",
    ]
)
def config(request) -> Config:
    cfg_str = request.getfixturevalue(request.param)
    config = Config().from_str(cfg_str)
    return config


@pytest.fixture
def nlp(config: Config) -> Language:
    nlp = assemble_from_config(config)
    return nlp


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_ner_config(config: Config):
    nlp = assemble_from_config(config)
    assert nlp.pipe_names == ["llm"]

    # also test nlp config from a dict in add_pipe
    component_cfg = dict(config["components"]["llm"])
    component_cfg.pop("factory")

    nlp2 = spacy.blank("en")
    nlp2.add_pipe("llm", config=component_cfg)
    assert nlp2.pipe_names == ["llm"]

    pipe = nlp.get_pipe("llm")
    assert isinstance(pipe, LLMWrapper)
    assert isinstance(pipe.task, ShardingLLMTask)

    labels = config["components"]["llm"]["task"]["labels"]
    labels = split_labels(labels)
    task = pipe.task
    assert isinstance(task, LabeledTask)
    assert sorted(task.labels) == sorted(tuple(labels))
    assert pipe.labels == task.labels
    assert nlp.pipe_labels["llm"] == list(task.labels)


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_str",
    ["fewshot_cfg_string_v3_lds", "fewshot_cfg_string_v3"],
)
@pytest.mark.parametrize(
    "text,gold_ents",
    [
        (
            "Marc and Bob both live in Ireland.",
            [("Marc", "PER"), ("Bob", "PER"), ("Ireland", "LOC")],
        ),
    ],
)
def test_ner_predict(cfg_str, text, gold_ents, request):
    """Use OpenAI to get NER results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    config = Config().from_str(request.getfixturevalue(cfg_str))

    nlp = spacy.util.load_model_from_config(config, auto_fill=True)
    doc = nlp(text)

    assert len(doc.ents) == len(gold_ents)
    for pred_ent, gold_ent in zip(doc.ents, gold_ents):
        assert (
            gold_ent[0] in pred_ent.text
        )  # occassionally, the LLM predicts "in Ireland" instead of just "Ireland"
        assert pred_ent.label_ in gold_ent[1].split("|")


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "text,gold_ents",
    [
        (
            "Marc and Bob both live in Ireland.",
            [("Marc", "PER"), ("Bob", "PER"), ("Ireland", "LOC")],
        ),
    ],
)
def test_llm_ner_predict(text, gold_ents):
    """Use llm_ner factory with default OpenAI model to get NER results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    nlp = spacy.blank("en")
    llm = nlp.add_pipe("llm_ner")
    for ent_str, ent_label in gold_ents:
        llm.add_label(ent_label)
    doc = nlp(text)

    assert len(doc.ents) == len(gold_ents)
    for pred_ent, gold_ent in zip(doc.ents, gold_ents):
        assert (
            gold_ent[0] in pred_ent.text
        )  # occassionally, the LLM predicts "in Ireland" instead of just "Ireland"
        assert pred_ent.label_ in gold_ent[1].split("|")


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_ner_io(nlp: Language):
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
    "response,normalizer,gold_ents",
    [
        (
            "1. Jean Jacques | True | PER | is a person's name\n"
            "2. Jaime | True | PER | is a person's name\n",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | PER | is a person's name\n"
            "2. Jaime | True | PER | is a person's name\n",
            strip_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | PER | is a person's name\n"
            "2. Jaime | True | PER | is a person's name\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is a person's name\n"
            "2. Jaime | True | per | is a person's name\n",
            strip_normalizer(),
            [],
        ),
        (
            "1. Jean Jacques | True | per | is a person's name\n"
            "2. Jaime | True | per | is a person's name\n",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is a person's name\n"
            "2. Jaime | True | PER | is a person's name\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is a person's name\n"
            "2. Jaime | True | per | is a person's name\n"
            "3. library | True | Org | is a organization\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "ORG")],
        ),
        (
            "1. Jean Jacques | True | per | is a person's name\n"
            "2. Jaime | True | per | is a person's name\n"
            "3. Jaime | True | RANDOM | is an entity\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
    ],
)
def test_ner_labels(
    response: str, normalizer: Callable[[str], str], gold_ents: List[Tuple[str, str]]
):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"

    llm_ner = make_ner_task_v3(examples=[], labels=labels, normalizer=normalizer)
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([[doc_in]], [[response]]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


@pytest.mark.parametrize(
    "response,alignment_mode,gold_ents",
    [
        (
            "1. Jacq | True | PER | is a person's name",
            "strict",
            [],
        ),
        (
            "1. Jacq | True | PER | is a person's name",
            "contract",
            [],
        ),
        (
            "1. Jacq | True | PER | is a person's name",
            "expand",
            [("Jacques", "PER")],
        ),
        (
            "1. Jean J | True | PER | is a person's name",
            "contract",
            [("Jean", "PER")],
        ),
        (
            "1. Jean Jacques | True | PER | is a person's name",
            "strict",
            [("Jean Jacques", "PER")],
        ),
        (
            "1. random | True | PER | is a person's name",
            "expand",
            [],
        ),
    ],
    ids=["strict_1", "contract_1", "expand_1", "strict_2", "contract_2", "expand_2"],
)
def test_ner_alignment(
    response: str,
    alignment_mode: Literal["strict", "contract", "expand"],
    gold_ents: List[Tuple[str, str]],
):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v3(
        examples=[], labels=labels, alignment_mode=alignment_mode
    )
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([[doc_in]], [[response]]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


def test_invalid_alignment_mode():
    labels = "PER,ORG,LOC"
    with pytest.raises(ValueError, match="Unsupported alignment mode 'invalid"):
        make_ner_task_v3(examples=[], labels=labels, alignment_mode="invalid")  # type: ignore


@pytest.mark.parametrize(
    "response, case_sensitive, gold_ents",
    [
        (
            "1. Jean | True | PER | is a person's name",
            False,
            [("jean", "PER")],
        ),
        (
            "1. Jean | True | PER | is a person's name",
            True,
            [("Jean", "PER")],
        ),
        (
            "1. jean | True | PER | is a person's name\n"
            "2. Jean | True | PER | is a person's name\n"
            "3. Jean Foundation | True | ORG | is the name of an Organization name",
            False,
            [("jean", "PER"), ("Jean", "PER"), ("Jean Foundation", "ORG")],
        ),
    ],
    ids=[
        "single_ent_case_insensitive",
        "single_ent_case_sensitive",
        "multiple_ents_case_insensitive",
    ],
)
def test_ner_matching(
    response: str, case_sensitive: bool, gold_ents: List[Tuple[str, str]]
):
    text = "This guy jean (or Jean) is the president of the Jean Foundation."
    labels = "PER,ORG,LOC"
    llm_ner = make_ner_task_v3(
        examples=[], labels=labels, case_sensitive_matching=case_sensitive
    )
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_ner.parse_responses([[doc_in]], [[response]]))[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc_out.ents]
    assert pred_ents == gold_ents


def test_jinja_template_rendering_without_examples():
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")
    llm_ner = make_ner_task_v3(labels=labels)
    prompt = list(llm_ner.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system.
Your task is to accept Text as input and extract named entities.
Entities must have one of the following labels: LOC, ORG, PER.
If a span is not an entity label it: `==NONE==`.


Here is an example of the output format for a paragraph using different labels than this task requires.
Only use this output format but use the labels provided
above instead of the ones defined in the example below.
Do not output anything besides entities in this output format.
Output entities in the order they occur in the input paragraph regardless of label.

Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

Paragraph: Sriracha sauce goes really well with hoisin stir fry, but you should add it after you use the wok.
Answer:
1. Sriracha sauce | True | INGREDIENT | is an ingredient to add to a stir fry
2. really well | False | ==NONE== | is a description of how well sriracha sauce goes with hoisin stir fry
3. hoisin stir fry | True | DISH | is a dish with stir fry vegetables and hoisin sauce
4. wok | True | EQUIPMENT | is a piece of cooking equipment used to stir fry ingredients

Paragraph: Alice and Bob went to the supermarket
Answer:
""".strip()
    )


@pytest.mark.parametrize("examples_file", ["ner.json", "ner.yml", "ner.jsonl"])
def test_jinja_template_rendering_with_examples(examples_dir: Path, examples_file: str):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """

    labels = "PER,ORG,LOC"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")
    examples = fewshot_reader(examples_dir / examples_file)
    llm_ner = make_ner_task_v3(examples=examples, labels=labels)
    prompt = list(llm_ner.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system.
Your task is to accept Text as input and extract named entities.
Entities must have one of the following labels: LOC, ORG, PER.
If a span is not an entity label it: `==NONE==`.

Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

Paragraph: Jack and Jill went up the hill.
Answer:
1. Jack | True | PER | is the name of a person
2. Jill | True | PER | is the name of a person
3. went up | False | ==NONE== | is a verb
4. hill | True | LOC | is a location

Paragraph: Alice and Bob went to the supermarket
Answer:
""".strip()
    )


@pytest.mark.parametrize("examples_file", ["ner.json", "ner.yml", "ner.jsonl"])
def test_jinja_template_rendering_with_label_definitions(
    examples_dir: Path, examples_file: str
):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")
    examples = fewshot_reader(examples_dir / examples_file)
    llm_ner = make_ner_task_v3(
        examples=examples,
        labels=labels,
        label_definitions={
            "PER": "Person definition",
            "ORG": "Organization definition",
            "LOC": "Location definition",
        },
    )
    prompt = list(llm_ner.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system.
Your task is to accept Text as input and extract named entities.
Entities must have one of the following labels: LOC, ORG, PER.
If a span is not an entity label it: `==NONE==`.

Below are definitions of each label to help aid you in what kinds of named entities to extract for each label.
Assume these definitions are written by an expert and follow them closely.
PER: Person definition
ORG: Organization definition
LOC: Location definition

Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

Paragraph: Jack and Jill went up the hill.
Answer:
1. Jack | True | PER | is the name of a person
2. Jill | True | PER | is the name of a person
3. went up | False | ==NONE== | is a verb
4. hill | True | LOC | is a location

Paragraph: Alice and Bob went to the supermarket
Answer:
""".strip()
    )


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (
            {
                "text": "I'm a wrong example. Entities should be a dict, not a list",
                # Should be: {"PER": ["Entities"], "ORG": ["dict", "list"]}
                "entities": [("PER", ("Entities")), ("ORG", ("dict", "list"))],
            },
            ValidationError,
        ),
        (
            {
                "text": "Jack is a name",
                "spans": [
                    {
                        "text": "Jack",
                        "is_entity": True,
                        "label": "PER",
                        "reason": "is a person's name",
                    }
                ],
            },
            NERTask,
        ),
    ],
)
def test_fewshot_example_data(value: dict, expected_type: type):
    with make_tempdir() as tmpdir:
        tmp_path = tmpdir / "wrong_example.yml"
        srsly.write_yaml(tmp_path, [value])

        try:
            task = make_ner_task_v3(
                examples=fewshot_reader(tmp_path), labels=["PER", "ORG", "LOC"]
            )
        except (ValidationError, ValueError) as e:
            assert type(e) == expected_type
        else:
            assert type(task) == expected_type


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "ner.jinja2")
    template = file_reader(template_path)
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    llm_ner = make_ner_task_v3(examples=[], labels=labels, template=template)
    prompt = list(llm_ner.generate_prompts([doc]))[0][0][0]
    assert prompt.strip().startswith("Here's the test template for the tests and stuff")


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("n_detections", [0, 1, 2])
def test_ner_scoring(fewshot_cfg_string_v3: str, n_detections: int):
    config = Config().from_str(fewshot_cfg_string_v3)
    nlp = assemble_from_config(config)

    examples = []
    for text in ["Alice works with Bob.", "Bob lives with Alice."]:
        predicted = nlp.make_doc(text)
        reference = nlp.make_doc(text)
        ent1 = Span(reference, 0, 1, label="PER")
        ent2 = Span(reference, 3, 4, label="PER")
        reference.set_ents([ent1, ent2][:n_detections])
        examples.append(Example(predicted, reference))

    scores = nlp.evaluate(examples)
    assert scores["ents_p"] == n_detections / 2
    assert scores["ents_r"] == (1 if n_detections != 0 else 0)
    assert scores["ents_f"] == (
        pytest.approx(0.666666666) if n_detections == 1 else n_detections / 2
    )


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_ner_init(noop_config: str, n_prompt_examples: int):
    config = Config().from_str(noop_config)
    config["components"]["llm"]["task"]["labels"] = ["PER", "LOC"]
    config["components"]["llm"]["task"]["examples"] = []
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp = assemble_from_config(config)

    examples = []
    for text in [
        "Alice works with Bob in London.",
        "Bob lives with Alice in Manchester.",
    ]:
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        reference.set_ents(
            [
                Span(reference, 0, 1, label="PER"),
                Span(reference, 3, 4, label="PER"),
                Span(reference, 5, 6, label="LOC"),
            ]
        )
        examples.append(Example(predicted, reference))

    task = cast(NERTask, nlp.get_pipe("llm").task)
    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }
    nlp.initialize(lambda: examples)

    assert set(task._label_dict.values()) == {"PER", "LOC"}

    if n_prompt_examples == -1:
        assert len(task._prompt_examples) == len(examples)
    else:
        assert len(task._prompt_examples) == n_prompt_examples

    if n_prompt_examples > 0:
        for eg in task._prompt_examples:
            prompt_example_labels = {ent.label for ent in eg.spans}
            if "==NONE==" not in prompt_example_labels:
                prompt_example_labels.add("==NONE==")
            assert prompt_example_labels == {"==NONE==", "PER", "LOC"}


def test_ner_serde(noop_config: str):
    config = Config().from_str(noop_config)

    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp1 = assemble_from_config(config)
        nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "per": "PER"}

    task1 = cast(NERTask, nlp1.get_pipe("llm").task)
    task2 = cast(NERTask, nlp2.get_pipe("llm").task)

    # Artificially add labels to task1
    task1._label_dict = labels
    task2._label_dict = {}

    assert task1._label_dict == labels
    assert task2._label_dict == dict()

    b = nlp1.to_bytes()
    nlp2.from_bytes(b)

    assert task1._label_dict == task2._label_dict == labels


def test_ner_to_disk(noop_config: str, tmp_path: Path):
    config = Config().from_str(noop_config)
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp1 = assemble_from_config(config)
        nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "org": "ORG", "per": "PER"}

    task1 = cast(NERTask, nlp1.get_pipe("llm").task)
    task2 = cast(NERTask, nlp2.get_pipe("llm").task)

    # Artificially add labels to task1
    task1._label_dict = labels
    task2._label_dict = {}

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


@pytest.mark.filterwarnings("ignore:Task supports sharding")
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
    @llm_tasks = "spacy.NER.v3"
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
    assert prompt_examples[0].spans == [
        SpanReason(
            text="Jack",
            is_entity=True,
            label="PERSON",
            reason="is the name of a person",
        ),
        SpanReason(
            text="Jill",
            is_entity=True,
            label="PERSON",
            reason="is the name of a person",
        ),
        SpanReason(
            text="went up", is_entity=False, label="==NONE==", reason="is a verb"
        ),
        SpanReason(
            text="hill", is_entity=True, label="LOCATION", reason="is a location"
        ),
    ]
    assert (
        prompt_examples[1].text
        == "Jack and Jill went up the hill and spaCy is a great tool."
    )
    assert prompt_examples[1].spans == [
        SpanReason(
            text="Jack",
            is_entity=True,
            label="PERSON",
            reason="is the name of a person",
        ),
        SpanReason(
            text="Jill",
            is_entity=True,
            label="PERSON",
            reason="is the name of a person",
        ),
        SpanReason(
            text="went up", is_entity=False, label="==NONE==", reason="is a verb"
        ),
        SpanReason(
            text="hill", is_entity=True, label="LOCATION", reason="is a location"
        ),
    ]


@pytest.mark.parametrize(
    "text, response, gold_ents",
    [
        (
            "The woman Paris was walking around in Paris, talking to her friend Paris",
            "1. Paris | True | PER | is the name of the woman\n"
            "2. Paris | True | LOC | is a city in France\n"
            "3. Paris | True | PER | is the name of the woman\n",
            [("Paris", "PER"), ("Paris", "LOC"), ("Paris", "PER")],
        ),
        (
            "Walking around Paris as a woman named Paris is quite a trip.",
            "1. Paris | True | LOC | is a city in France\n"
            "2. Paris | True | PER | is the name of the woman\n",
            [("Paris", "LOC"), ("Paris", "PER")],
        ),
    ],
    ids=["3_ents", "2_ents"],
)
def test_regression_span_task_response_parse(
    text: str, response: str, gold_ents: List[Tuple[str, str]]
):
    """Test based on spaCy issue: https://github.com/explosion/spaCy/discussions/12812
    where parsing wasn't working for NER when the same text could map to 2 labels.
    In the user's case "Paris" could be a person's name or a location.
    """

    nlp = spacy.blank("en")
    example_doc = nlp.make_doc(text)
    ner_task = make_ner_task_v3(examples=[], labels=["PER", "LOC"])
    span_reasons = _extract_span_reasons_cot(ner_task, response)
    assert len(span_reasons) == len(gold_ents)

    docs = list(ner_task.parse_responses([[example_doc]], [[response]]))
    assert len(docs) == 1

    doc = docs[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
    assert pred_ents == gold_ents


@pytest.mark.parametrize(
    "text, response, gold_ents",
    [
        (
            "FooBar, Inc. is a large organization in the U.S.",
            "1. FooBar, Inc. | True | ORG | is the name of an organization\n"
            "2. U.S. | True | LOC | is a country\n",
            [("FooBar, Inc.", "ORG"), ("U.S.", "LOC")],
        ),
    ],
)
def test_regression_span_task_comma(
    text: str, response: str, gold_ents: List[Tuple[str, str]]
):
    """Test that spacy.NER.v3 can deal with comma's in entities"""

    nlp = spacy.blank("en")
    example_doc = nlp.make_doc(text)
    ner_task = make_ner_task_v3(examples=[], labels=["ORG", "LOC"])
    span_reasons = _extract_span_reasons_cot(ner_task, response)
    assert len(span_reasons) == len(gold_ents)
    docs = list(ner_task.parse_responses([[example_doc]], [[response]]))
    assert len(docs) == 1
    doc = docs[0]
    pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
    assert pred_ents == gold_ents


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_add_label():
    nlp = spacy.blank("en")
    llm = nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v3",
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v1",
            },
        },
    )

    nlp.initialize()
    text = "Jack and Jill visited France."
    doc = nlp(text)
    assert len(doc.ents) == 0

    for label, definition in [
        ("PERSON", "Every person with the name Jack"),
        ("LOCATION", "A geographical location, like a country or a city"),
        ("COMPANY", None),
    ]:
        llm.add_label(label, definition)
    doc = nlp(text)
    assert len(doc.ents) > 1


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_clear_label():
    nlp = spacy.blank("en")
    llm = nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v3",
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v1",
            },
        },
    )

    nlp.initialize()
    text = "Jack and Jill visited France."
    doc = nlp(text)

    for label in ["PERSON", "LOCATION"]:
        llm.add_label(label)
    doc = nlp(text)
    assert len(doc.ents) == 3

    llm.clear()

    doc = nlp(text)
    assert len(doc.ents) == 0
