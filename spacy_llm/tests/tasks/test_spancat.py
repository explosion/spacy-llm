from pathlib import Path
from typing import cast

import pytest
import spacy
from confection import Config
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, lowercase_normalizer, strip_normalizer
from spacy_llm.tasks import make_spancat_task_v3
from spacy_llm.tasks.spancat import SpanCatTask
from spacy_llm.tasks.util import find_substrings
from spacy_llm.ty import Labeled, LLMTask
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
    @llm_tasks = "spacy.SpanCat.v3"
    description = "This is a description"
    labels = ["PER", "ORG", "LOC"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "spancat.json"))}

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "spancat.jinja2"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    """


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["fewshot_cfg_string", "zeroshot_cfg_string"])
def test_spancat_config(cfg_string, request):
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]

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
@pytest.mark.parametrize("cfg_string", ["", "fewshot_cfg_string"])
def test_spancat_predict(cfg_string, request):
    """Use OpenAI to get zero-shot NER results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)
    assert len(doc.spans["sc"]) > 0
    for ent in doc.spans["sc"]:
        assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.external
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_spancat_io(cfg_string, request):
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
    assert len(doc.spans["sc"]) > 0
    for ent in doc.spans["sc"]:
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
    "text,response,gold_spans",
    [
        # simple
        (
            "Jean Jacques and Jaime went to the library.",
            "PER: Jean Jacques, Jaime\nLOC: library",
            [("Jean Jacques", "PER"), ("Jaime", "PER"), ("library", "LOC")],
        ),
        # overlapping: should only return all spans
        (
            "The Manila Observatory was founded in 1865.",
            "LOC: The Manila Observatory, Manila, Manila Observatory",
            [
                ("The Manila Observatory", "LOC"),
                ("Manila", "LOC"),
                ("Manila Observatory", "LOC"),
            ],
        ),
        # flipped: order shouldn't matter
        (
            "Take the road from Downtown and turn left at the public market.",
            "LOC: public market, Downtown",
            [("Downtown", "LOC"), ("public market", "LOC")],
        ),
    ],
)
def test_spancat_zero_shot_task(text, response, gold_spans):
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(examples=[], labels=labels)
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list so we get what's inside
    doc_out = list(llm_spancat.parse_responses([doc_in], [response]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


@pytest.mark.parametrize(
    "response,normalizer,gold_spans",
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
def test_spancat_labels(response, normalizer, gold_spans):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, normalizer=normalizer
    )
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([doc_in], [response]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


@pytest.mark.parametrize(
    "response,alignment_mode,gold_spans",
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
def test_spancat_alignment(response, alignment_mode, gold_spans):
    text = "Jean Jacques and Jaime went to the library."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, alignment_mode=alignment_mode
    )
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([doc_in], [response]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


def test_invalid_alignment_mode():
    labels = "PER,ORG,LOC"
    with pytest.raises(ValueError, match="Unsupported alignment mode 'invalid"):
        make_spancat_task_v3(examples=[], labels=labels, alignment_mode="invalid")


@pytest.mark.parametrize(
    "response,case_sensitive,single_match,gold_spans",
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
def test_spancat_matching(response, case_sensitive, single_match, gold_spans):
    text = "This guy jean (or Jean) is the president of the Jean Foundation."
    labels = "PER,ORG,LOC"
    llm_spancat = make_spancat_task_v3(
        examples=[], labels=labels, case_sensitive_matching=case_sensitive
    )
    # Prepare doc
    nlp = spacy.blank("xx")
    doc_in = nlp.make_doc(text)
    # Pass to the parser
    # Note: parser() returns a list
    doc_out = list(llm_spancat.parse_responses([doc_in], [response]))[0]
    pred_spans = [(span.text, span.label_) for span in doc_out.spans["sc"]]
    assert pred_spans == gold_spans


@pytest.mark.parametrize(
    "examples_file", ["spancat.json", "spancat.yml", "spancat.jsonl"]
)
def test_jinja_template_rendering_with_examples(examples_dir: Path, examples_file: str):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    labels = "PER,ORG,LOC,DESTINATION"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc("Alice and Bob went to the supermarket")

    examples = fewshot_reader(examples_dir / examples_file)
    llm_spancat = make_spancat_task_v3(examples=examples, labels=labels)
    prompt = list(llm_spancat.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
You are an expert Named Entity Recognition (NER) system.
Your task is to accept Text as input and extract named entities.
The entities you extract can overlap with each other.

Entities must have one of these labels: PER, ORG, LOC, DESTINATION.
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


@pytest.mark.parametrize("n_detections", [0, 1, 2])
def test_spancat_scoring(noop_config, n_detections):
    config = Config().from_str(noop_config)
    nlp = assemble_from_config(config)

    examples = []

    for text in ["Alice works with Bob.", "Bob lives with Alice."]:
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        reference.spans["sc"] = [
            Span(reference, 0, 1, label="PER"),
            Span(reference, 3, 4, label="PER"),
        ][:n_detections]

        examples.append(Example(predicted, reference))

    scores = nlp.evaluate(examples)

    assert scores["spans_sc_p"] == n_detections / 2


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_spancat_init(noop_config: str, n_prompt_examples: bool):
    config = Config().from_str(noop_config)
    config["components"]["llm"]["task"]["labels"] = ["PER", "LOC", "DESTINATION"]
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
        assert len(task._prompt_examples) == 3
    elif n_prompt_examples == 0:
        assert len(task._prompt_examples) == 1
    elif n_prompt_examples in (1, 2):
        assert len(task._prompt_examples) == n_prompt_examples

    if n_prompt_examples > 0:
        for eg in task._prompt_examples:
            prompt_example_labels = {ent.label for ent in eg.spans}
            if "==NONE==" not in prompt_example_labels:
                prompt_example_labels.add("==NONE==")
            assert prompt_example_labels == {"==NONE==", "PER", "LOC", "DESTINATION"}


def test_spancat_serde(noop_config):
    config = Config().from_str(noop_config)

    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)

    labels = {"loc": "LOC", "per": "PER"}

    task1 = cast(SpanCatTask, nlp1.get_pipe("llm").task)
    task2 = cast(SpanCatTask, nlp2.get_pipe("llm").task)

    # Artificially add labels to task1
    task1._label_dict = labels

    assert task1._label_dict == labels
    assert task2._label_dict == dict()

    b = nlp1.to_bytes()
    nlp2.from_bytes(b)

    assert task1._label_dict == task2._label_dict == labels
