from pathlib import Path

import pytest
import spacy
from confection import Config
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.registry import fewshot_reader, file_reader
from spacy_llm.tasks.lemma import LemmaTask
from spacy_llm.util import assemble_from_config

from ...tasks import make_lemma_task
from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


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
    @llm_tasks = "spacy.Lemma.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    """


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
    @llm_tasks = "spacy.Lemma.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {"temperature": 0}
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
    @llm_tasks = "spacy.Lemma.v1"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "lemma.yml"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}
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
    @llm_tasks = "spacy.Lemma.v1"

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "lemma.jinja2"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}
    """


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
        "ext_template_cfg_string",
    ],
)
def test_lemma_config(cfg_string, request):
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


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
        "ext_template_cfg_string",
    ],
)
def test_lemma_predict(cfg_string, request):
    """Use OpenAI to get zero-shot LEMMA results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    lemmas = [str(token.lemma_) for token in nlp("I've watered the plants.")]
    # Compare lemmas for correctness, if we are not using the external dummy template.
    if cfg_string != "ext_template_cfg_string":
        assert lemmas in (
            ["-PRON-", "have", "water", "the", "plant", "."],
            ["I", "have", "water", "the", "plant", "."],
        )


@pytest.mark.external
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
    ],
)
def test_lemma_io(cfg_string, request):
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    lemmas = [str(token.lemma_) for token in nlp2("I've watered the plants.")]
    if cfg_string != "ext_template_cfg_string":
        assert lemmas in (
            ["-PRON-", "have", "water", "the", "plant", "."],
            ["I", "have", "water", "the", "plant", "."],
        )


def test_jinja_template_rendering_without_examples():
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "Alice and Bob went to the supermarket"
    doc = nlp.make_doc(text)

    lemma_task = make_lemma_task(examples=None)
    prompt = list(lemma_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == f"""
You are an expert lemmatization system. Your task is to accept Text as input and identify the lemma for every token in the Text.
Consider that contractions represent multiple words. Each word in a contraction should be annotated with its lemma separately.
Output each original word on a new line, followed by a colon and the word's lemma - like this:
'''
Word1: Lemma of Word1
Word2: Lemma of Word2
'''
Include the final punctuation token in this list.
Prefix with your output with "Lemmatized text".


Here is the text that needs to be lemmatized:
'''
{text}
'''
""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "lemma.json"),
        str(EXAMPLES_DIR / "lemma.yml"),
        str(EXAMPLES_DIR / "lemma.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "Alice and Bob went to the supermarket."
    doc = nlp.make_doc(text)

    lemma_task = make_lemma_task(examples=fewshot_reader(examples_path))
    prompt = list(lemma_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == f"""
You are an expert lemmatization system. Your task is to accept Text as input and identify the lemma for every token in the Text.
Consider that contractions represent multiple words. Each word in a contraction should be annotated with its lemma separately.
Output each original word on a new line, followed by a colon and the word's lemma - like this:
'''
Word1: Lemma of Word1
Word2: Lemma of Word2
'''
Include the final punctuation token in this list.
Prefix with your output with "Lemmatized text".

Below are some examples (only use these as a guide):

Text:
'''
The arc of the moral universe is long, but it bends toward justice.
'''
Lemmas:
'''
The: The
arc: arc
of: of
the: the
moral: moral
universe: universe
is: be
long: long
,: ,
but: but
it: it
bends: bend
toward: toward
justice: justice
.: .
'''

Text:
'''
Life can only be understood backwards; but it must be lived forwards.
'''
Lemmas:
'''
Life: Life
can: can
only: only
be: be
understood: understand
backwards: backwards
;: ;
but: but
it: it
must: must
be: be
lived: lived
forwards: forwards
.: .
'''

Text:
'''
I'm buying ice cream.
'''
Lemmas:
'''
I: I
'm: be
buying: buy
ice: ice
cream: cream
.: .
'''

Here is the text that needs to be lemmatized:
'''
{text}
'''
""".strip()
    )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "lemma.jinja2")
    template = file_reader(template_path)
    text = "Alice and Bob went to the supermarket"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(text)

    lemma_task = make_lemma_task(template=template)
    prompt = list(lemma_task.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == f"""
This is a test LEMMA template.
Here is the text: {text}
""".strip()
    )


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_lemma_init(noop_config, n_prompt_examples: int):
    config = Config().from_str(noop_config)
    nlp = assemble_from_config(config)

    examples = []
    pred_words_1 = ["Alice", "works", "all", "evenings"]
    gold_lemmas_1 = ["Alice", "work", "all", "evening"]
    pred_1 = Doc(nlp.vocab, words=pred_words_1)
    gold_1 = Doc(nlp.vocab, words=pred_words_1, lemmas=gold_lemmas_1)
    examples.append(Example(pred_1, gold_1))

    pred_words_2 = ["Bob", "loves", "living", "cities"]
    gold_lemmas_2 = ["Bob", "love", "live", "city"]
    pred_2 = Doc(nlp.vocab, words=pred_words_2)
    gold_2 = Doc(nlp.vocab, words=pred_words_2, lemmas=gold_lemmas_2)
    examples.append(Example(pred_2, gold_2))

    _, llm = nlp.pipeline[0]
    task: LemmaTask = llm._task

    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }
    nlp.initialize(lambda: examples)

    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(examples)
