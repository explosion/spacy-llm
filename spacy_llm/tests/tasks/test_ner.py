# mypy: ignore-errors
from pathlib import Path

import pytest
import spacy
import srsly
from confection import Config
from pydantic import ValidationError
from spacy.util import make_tempdir

from spacy_llm.registry import strip_normalizer, lowercase_normalizer, fewshot_reader
from spacy_llm.tasks.ner import find_substrings, NERTask
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
    @llm_tasks = "spacy.NER.v1"
    labels = PER,ORG,LOC

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
    @llm_tasks = "spacy.NER.v1"
    labels = PER,ORG,LOC

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "ner_examples.yml"))}

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"
    config: {{}}
    """


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["fewshot_cfg_string"])  # "zeroshot_cfg_string",
def test_ner_config(cfg_string, request):
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]


@pytest.mark.external
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_ner_predict(cfg_string, request):
    """Use OpenAI to get zero-shot NER results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    text = "Marc and Bob both live in Ireland."
    doc = nlp(text)
    assert len(doc.ents) > 0
    for ent in doc.ents:
        assert ent.label_ in ["PER", "ORG", "LOC"]


@pytest.mark.external
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
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
    assert len(doc.ents) > 0
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
    llm_ner = NERTask(labels=labels)
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
    llm_ner = NERTask(labels=labels, normalizer=normalizer)
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
    llm_ner = NERTask(labels=labels, alignment_mode=alignment_mode)
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
        NERTask(labels=labels, alignment_mode="invalid")


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
    llm_ner = NERTask(
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

    llm_ner = NERTask(labels=labels, examples=None)
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
From the text below, extract the following entities in the following format:
PER: <comma delimited list of strings>
ORG: <comma delimited list of strings>
LOC: <comma delimited list of strings>

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
        str(EXAMPLES_DIR / "ner_examples.json"),
        str(EXAMPLES_DIR / "ner_examples.yml"),
        str(EXAMPLES_DIR / "ner_examples.jsonl"),
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
    llm_ner = NERTask(labels=labels, examples=examples)
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
From the text below, extract the following entities in the following format:
PER: <comma delimited list of strings>
ORG: <comma delimited list of strings>
LOC: <comma delimited list of strings>

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

        with pytest.raises(ValidationError):
            NERTask(labels="PER,ORG,LOC", examples=fewshot_reader(tmp_path))
