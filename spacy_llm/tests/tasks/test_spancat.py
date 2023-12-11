from pathlib import Path
from typing import Callable, List, Tuple, cast

import pytest
import spacy
from confection import Config
from spacy.language import Language
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, lowercase_normalizer, strip_normalizer
from spacy_llm.tasks import make_spancat_task_v3
from spacy_llm.tasks.spancat import SpanCatTask
from spacy_llm.tasks.util import find_substrings
from spacy_llm.ty import LabeledTask, ShardingLLMTask
from spacy_llm.util import assemble_from_config, split_labels

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture
def examples_dir():
    return EXAMPLES_DIR


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
    @llm_tasks = "spacy.SpanCat.v3"
    labels = ["PER", "ORG", "LOC", "DESTINATION"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "spancat.yml"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    output = 1. Bob | True | PER | is the name of a person
        2. Alice | True | PER | is the name of a person
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
    @llm_tasks = "spacy.SpanCat.v3"
    labels = ["PER", "ORG", "LOC", "DESTINATION"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "spancat.yml"))}

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
    @llm_tasks = "spacy.SpanCat.v3"
    description = "This is a description"
    labels = ["PER", "ORG", "LOC", "DESTINATION"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "spancat.json"))}

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "spancat.jinja2"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v2"
    """


@pytest.fixture(
    params=[
        "fewshot_cfg_string",
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
def test_spancat_config(config: Config):
    nlp = assemble_from_config(config)
    assert nlp.pipe_names == ["llm"]

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
def test_spancat_predict(nlp: Language):
    """Use OpenAI to get zero-shot spancat results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)
    assert len(doc.spans["sc"]) > 0
    for ent in doc.spans["sc"]:
        assert ent.label_ in ["PER", "ORG", "LOC", "DESTINATION"]


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
def test_spancat_io(nlp: Language):
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    text = "Marc and Bob both live in Ireland."
    doc = nlp2(text)
    assert len(doc.spans["sc"]) > 0
    for ent in doc.spans["sc"]:
        assert ent.label_ in ["PER", "ORG", "LOC", "DESTINATION"]


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
    "text,response,gold_spans",
    [
        # simple
        (
            "Jean Jacques and Jaime went to the library.",
            "1. Jean Jacques | True | PER | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n"
            "3. library | True | LOC | is a place you can go to with lots of books\n",
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "LOC")],
        ),
        # overlapping: should only return all spans
        (
            "The Manila Observatory was founded in 1865.",
            "1. The Manila Observatory | True | LOC | is a place\n"
            "2. Manila | True | LOC | is a city\n"
            "3. Manila Observatory | True | LOC | is a place\n",
            [
                ("The Manila Observatory", "LOC"),
                ("Manila", "LOC"),
                ("Manila Observatory", "LOC"),
            ],
        ),
    ],
    ids=["simple", "overlapping"],
)
def test_spancat_matching_shot_task(text: str, response: str, gold_spans):
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(examples=[], labels=labels)
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list so we get what's inside
    doc_out = list(llm_spancat.parse_responses([[doc_in]], [[response]]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


@pytest.mark.parametrize(
    "response,normalizer,gold_spans",
    [
        (
            "1. Jean Jacques | True | PER | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | PER | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n",
            strip_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | PER | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is the name of a person\n"
            "2. Jaime | True | per | is the name of a person\n",
            strip_normalizer(),
            [],
        ),
        (
            "1. Jean Jacques | True | PER | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n",
            None,
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is the name of a person\n"
            "2. Jaime | True | PER | is the name of a person\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
        (
            "1. Jean Jacques | True | per | is the name of a person\n"
            "2. Jaime | True | per | is the name of a person\n"
            "3. library | True | Org | is an organization\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "ORG")],
        ),
        (
            "1. Jean Jacques | True | per | is the name of a person\n"
            "2. Jaime | True | per | is the name of a person\n"
            "3. library | True | RANDOM | is an organization\n",
            lowercase_normalizer(),
            [("Jean Jacques", "PER"), ("Jaime", "PER")],
        ),
    ],
)
def test_spancat_labels(
    response: str, normalizer: Callable[[str], str], gold_spans: List[Tuple[str, str]]
):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, normalizer=normalizer
    )
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([[doc_in]], [[response]]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


@pytest.mark.parametrize(
    "response,alignment_mode,gold_spans",
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
def test_spancat_alignment(response, alignment_mode, gold_spans):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, alignment_mode=alignment_mode
    )
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([[doc_in]], [[response]]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


def test_invalid_alignment_mode():
    labels = "PER,ORG,LOC"
    with pytest.raises(ValueError, match="Unsupported alignment mode 'invalid"):
        make_spancat_task_v3(examples=[], labels=labels, alignment_mode="invalid")


@pytest.mark.parametrize(
    "response, case_sensitive, gold_spans",
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
def test_spancat_matching(response, case_sensitive, gold_spans):
    text = "This guy jean (or Jean) is the president of the Jean Foundation."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, case_sensitive_matching=case_sensitive
    )
    # Prepare doc
    nlp = spacy.blank("en")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([[doc_in]], [[response]]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


def test_jinja_template_rendering_without_examples():
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")
    llm_spancat = make_spancat_task_v3(labels=labels)
    prompt = list(llm_spancat.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == """
You are an expert Entity Recognition system.
Your task is to accept Text as input and extract named entities.
The entities you extract can overlap with each other.

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


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "spancat.json"),
        str(EXAMPLES_DIR / "spancat.yml"),
        str(EXAMPLES_DIR / "spancat.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path: Path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC,DESTINATION"
    nlp = spacy.blank("en")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    examples = fewshot_reader(examples_path)
    llm_spancat = make_spancat_task_v3(labels=labels, examples=examples)
    prompt = list(llm_spancat.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == """
You are an expert Entity Recognition system.
Your task is to accept Text as input and extract named entities.
The entities you extract can overlap with each other.

Entities must have one of the following labels: DESTINATION, LOC, ORG, PER.
If a span is not an entity label it: `==NONE==`.


Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

Paragraph: Jack and Jill went up the hill.
Answer:
1. Jack | True | PER | is the name of a person
2. Jill | True | PER | is the name of a person
3. went up | False | ==NONE== | is a verb
4. hill | True | LOC | is a location
5. hill | True | DESTINATION | is a destination

Paragraph: Alice and Bob went to the supermarket
Answer:
""".strip()
    )


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("n_detections", [0, 1, 2])
def test_spancat_scoring(fewshot_cfg_string: str, n_detections: int):
    config = Config().from_str(fewshot_cfg_string)
    nlp = assemble_from_config(config)

    examples = []
    for text in ["Alice works with Bob.", "Bob lives with Alice."]:
        predicted = nlp.make_doc(text)
        reference = nlp.make_doc(text)
        ent1 = Span(reference, 0, 1, label="PER")
        ent2 = Span(reference, 3, 4, label="PER")
        reference.spans["sc"] = [ent1, ent2][:n_detections]
        examples.append(Example(predicted, reference))

    scores = nlp.evaluate(examples)
    assert scores["spans_sc_p"] == n_detections / 2
    assert scores["spans_sc_r"] == (1 if n_detections != 0 else 0)
    assert scores["spans_sc_f"] == (
        pytest.approx(0.666666666) if n_detections == 1 else n_detections / 2
    )


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_spancat_init(noop_config: str, n_prompt_examples: bool):
    config = Config().from_str(noop_config)
    config["components"]["llm"]["task"]["labels"] = ["PER", "LOC", "DESTINATION"]
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

        reference.spans["sc"] = [
            Span(reference, 0, 1, label="PER"),
            Span(reference, 3, 4, label="PER"),
            Span(reference, 5, 6, label="LOC"),
        ]

        examples.append(Example(predicted, reference))

    _, llm = nlp.pipeline[0]
    task: SpanCatTask = llm._task

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }

    nlp.initialize(lambda: examples)

    assert set(task._label_dict.values()) == {"PER", "LOC", "DESTINATION"}
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


def test_spancat_serde(noop_config):
    config = Config().from_str(noop_config)
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp1 = assemble_from_config(config)
        nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "per": "PER"}

    task1 = cast(SpanCatTask, nlp1.get_pipe("llm").task)
    task2 = cast(SpanCatTask, nlp2.get_pipe("llm").task)

    # Artificially add labels to task1
    task1._label_dict = labels
    task2._label_dict = dict()

    assert task1._label_dict == labels
    assert task2._label_dict == dict()

    b = nlp1.to_bytes()
    nlp2.from_bytes(b)

    assert task1._label_dict == task2._label_dict == labels
