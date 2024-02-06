from pathlib import Path

import pytest
import spacy
from confection import Config
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.registry import file_reader
from spacy_llm.util import assemble_from_config

from ...tasks import RawTask, make_raw_task
from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@pytest.fixture
def noop_config():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.Raw.v1"

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
    @llm_tasks = "spacy.Raw.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v3"
    """


@pytest.fixture
def fewshot_cfg_string():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.Raw.v1"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "raw.yml"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v3"
    """


@pytest.fixture
def ext_template_cfg_string():
    """Simple zero-shot config with an external template"""

    return f"""
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]
    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.Raw.v1"

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "raw.jinja2"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v3"
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
def test_raw_config(cfg_string, request):
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
def test_raw_predict(cfg_string, request):
    """Use OpenAI to get zero-shot LEMMA results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp("What's the weather like?")._.llm_reply


@pytest.mark.external
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
    ],
)
def test_raw_io(cfg_string, request):
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    assert nlp2("I've watered the plants.")._.llm_reply


def test_jinja_template_rendering_without_examples():
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("en")
    text = "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    doc = nlp.make_doc(text)

    raw_task = make_raw_task(examples=None)
    prompt = list(raw_task.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == f"""
Text:
{text}
Reply:
""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "raw.json"),
        str(EXAMPLES_DIR / "raw.yml"),
        str(EXAMPLES_DIR / "raw.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("en")
    text = "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    doc = nlp.make_doc(text)

    raw_task = make_raw_task(examples=None)
    prompt = list(raw_task.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == f"""
Text:
{text}
Reply:
""".strip()
    )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "raw.jinja2")
    template = file_reader(template_path)
    text = "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)

    raw_task = make_raw_task(examples=None, template=template)
    prompt = list(raw_task.generate_prompts([doc]))[0][0][0]

    assert (
        prompt.strip()
        == f"""
This is a test RAW template.
Here is the text: {text}
""".strip()
    )


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_raw_init(noop_config, n_prompt_examples: int):
    config = Config().from_str(noop_config)
    with pytest.warns(UserWarning, match="Task supports sharding"):
        nlp = assemble_from_config(config)

    examples = []
    text = "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    gold_1 = nlp.make_doc(text)
    gold_1._.llm_reply = "Plenty"
    examples.append(Example(nlp.make_doc(text), gold_1))

    text = "Who sells seashells by the seashore?"
    gold_2 = nlp.make_doc(text)
    gold_2._.llm_reply = "Shelly"
    examples.append(Example(nlp.make_doc(text), gold_2))

    _, llm = nlp.pipeline[0]
    task: RawTask = llm._task

    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }
    nlp.initialize(lambda: examples)

    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(examples)
