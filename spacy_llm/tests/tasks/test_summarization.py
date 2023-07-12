import re
from pathlib import Path

import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.registry import fewshot_reader, file_reader
from spacy_llm.ty import LLMTask
from spacy_llm.util import assemble_from_config

from ...tasks import make_summarization_task
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
    @llm_tasks = "spacy.Summarization.v1"
    max_n_words = 20

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
    @llm_tasks = "spacy.Summarization.v1"
    max_n_words = 20

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "summarization.yml"))}

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
    @llm_tasks = "spacy.Summarization.v1"
    max_n_words = 20

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "summarization.jinja2"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}
    """


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
    @llm_tasks = "spacy.Summarization.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"
    """


@pytest.fixture
def example_text() -> str:
    """Returns string to be used as example in tests."""
    return (
        "The atmosphere of Earth is the layer of gases, known collectively as air, retained by Earth's gravity "
        "that surrounds the planet and forms its planetary atmosphere. The atmosphere of Earth creates pressure, "
        "absorbs most meteoroids and ultraviolet solar radiation, warms the surface through heat retention "
        "(greenhouse effect), allowing life and liquid water to exist on the Earth's surface, and reduces "
        "temperature extremes between day and night (the diurnal temperature variation)."
    )


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
def test_summarization_config(cfg_string, request):
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
def test_summarization_predict(cfg_string, example_text, request):
    """Use OpenAI to get summarize text.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    orig_cfg_string = cfg_string
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)

    # One of the examples exceeds the set max_n_words, so we expect a warning to be emitted.
    if orig_cfg_string == "fewshot_cfg_string":
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "The provided example 'Life is a quality th...' has a summary of length 28, but `max_n_words` == 20."
            ),
        ):
            doc = nlp(example_text)
    else:
        doc = nlp(example_text)

    # Check whether a non-empty summary was written and we are somewhat close to the desired upper length limit.
    assert 0 < len(doc._.summary)
    if "ext" not in orig_cfg_string:
        nlp.select_pipes(disable=["llm"])
        assert (
            len(nlp(doc._.summary))
            <= orig_config["components"]["llm"]["task"]["max_n_words"] * 1.5
        )


# @pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string_and_field",
    [
        ("zeroshot_cfg_string", None),
        ("fewshot_cfg_string", None),
        ("ext_template_cfg_string", None),
        ("zeroshot_cfg_string", "summary_x"),
    ],
)
def test_summarization_io(cfg_string_and_field, example_text, request):
    cfg_string, field = cfg_string_and_field
    orig_cfg_string = cfg_string
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    if field:
        orig_config["components"]["llm"]["task"]["field"] = field
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]

    if orig_cfg_string == "fewshot_cfg_string":
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "The provided example 'Life is a quality th...' has a summary of length 28, but `max_n_words` == 20."
            ),
        ):
            doc = nlp2(example_text)
    else:
        doc = nlp2(example_text)

    field = "summary" if field is None else field
    nlp2.select_pipes(disable=["llm"])
    assert 0 < len(nlp2(getattr(doc._, field)))
    if "ext" not in orig_cfg_string:
        assert (
            len(nlp2(getattr(doc._, field)))
            <= orig_config["components"]["llm"]["task"]["max_n_words"] * 1.5
        )


def test_jinja_template_rendering_without_examples(example_text):
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(example_text)

    llm_ner = make_summarization_task(examples=None, max_n_words=10)
    prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == f"""
You are an expert summarization system. Your task is to accept Text as input and summarize the Text in a concise way.
The summary must not, under any circumstances, contain more than 10 words.
Here is the Text that needs to be summarized:
'''
{example_text}
'''
Summary:""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "summarization.json"),
        str(EXAMPLES_DIR / "summarization.yml"),
        str(EXAMPLES_DIR / "summarization.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path, example_text):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(example_text)

    examples = fewshot_reader(examples_path)
    llm_ner = make_summarization_task(examples=examples, max_n_words=20)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The provided example 'Life is a quality th...' has a summary of length 28, but `max_n_words` == 20."
        ),
    ):
        prompt = list(llm_ner.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == f"""
You are an expert summarization system. Your task is to accept Text as input and summarize the Text in a concise way.
The summary must not, under any circumstances, contain more than 20 words.
Below are some examples (only use these as a guide):

Text:
'''
The United Nations, referred to informally as the UN, is an intergovernmental organization whose stated purposes are to maintain international peace and security, develop friendly relations among nations, achieve international cooperation, and serve as a centre for harmonizing the actions of nations. It is the world's largest international organization. The UN is headquartered on international territory in New York City, and the organization has other offices in Geneva, Nairobi, Vienna, and The Hague, where the International Court of Justice is headquartered.

The UN was established after World War II with the aim of preventing future world wars, and succeeded the League of Nations, which was characterized as ineffective. On 25 April 1945, 50 nations met in San Francisco, California for a conference and started drafting the UN Charter, which was adopted on 25 June 1945. The charter took effect on 24 October 1945, when the UN began operations. The organization's objectives, as defined by its charter, include maintaining international peace and security, protecting human rights, delivering humanitarian aid, promoting sustainable development, and upholding international law. At its founding, the UN had 51 member states; as of 2023, it has 193 – almost all of the world's sovereign states.
'''
Summary:
'''
UN is an intergovernmental organization to foster international peace, security, and cooperation. Established after WW2 with 51 members, now 193.
'''

Text:
'''
Life is a quality that distinguishes matter that has biological processes, such as signaling and self-sustaining processes, from matter that does not, and is defined by the capacity for growth, reaction to stimuli, metabolism, energy transformation, and reproduction. Various forms of life exist, such as plants, animals, fungi, protists, archaea, and bacteria. Biology is the science that studies life.

The gene is the unit of heredity, whereas the cell is the structural and functional unit of life. There are two kinds of cells, prokaryotic and eukaryotic, both of which consist of cytoplasm enclosed within a membrane and contain many biomolecules such as proteins and nucleic acids. Cells reproduce through a process of cell division, in which the parent cell divides into two or more daughter cells and passes its genes onto a new generation, sometimes producing genetic variation.

Organisms, or the individual entities of life, are generally thought to be open systems that maintain homeostasis, are composed of cells, have a life cycle, undergo metabolism, can grow, adapt to their environment, respond to stimuli, reproduce and evolve over multiple generations. Other definitions sometimes include non-cellular life forms such as viruses and viroids, but they are usually excluded because they do not function on their own; rather, they exploit the biological processes of hosts.
'''
Summary:
'''
Life is a quality defined by biological processes, including reproduction, genetics, and metabolism. There are two types of cells and organisms that can grow, respond, reproduce, and evolve.
'''

Here is the Text that needs to be summarized:
'''
{example_text}
'''
Summary:
""".strip()
    )


def test_external_template_actually_loads(example_text):
    template_path = str(TEMPLATES_DIR / "summarization.jinja2")
    template = file_reader(template_path)
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(example_text)

    llm_ner = make_summarization_task(template=template)
    prompt = list(llm_ner.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == """
This is a test summarization template.
Here is the text: The atmosphere of Earth is the layer of gases, known collectively as air, retained by Earth's gravity that surrounds the planet and forms its planetary atmosphere. The atmosphere of Earth creates pressure, absorbs most meteoroids and ultraviolet solar radiation, warms the surface through heat retention (greenhouse effect), allowing life and liquid water to exist on the Earth's surface, and reduces temperature extremes between day and night (the diurnal temperature variation).
""".strip()
    )


def test_ner_serde(noop_config):
    config = Config().from_str(noop_config)
    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)
    nlp2.from_bytes(nlp1.to_bytes())


def test_ner_to_disk(noop_config, tmp_path: Path):
    config = Config().from_str(noop_config)
    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)

    path = tmp_path / "model"
    nlp1.to_disk(path)

    cfgs = list(path.rglob("cfg"))
    assert len(cfgs) == 1

    nlp2.from_disk(path)
