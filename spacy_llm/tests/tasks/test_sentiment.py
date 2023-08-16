from pathlib import Path

import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.registry import fewshot_reader, file_reader

from ...tasks import make_sentiment_task
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
    @llm_tasks = "spacy.Sentiment.v1"

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
    @llm_tasks = "spacy.Sentiment.v1"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "sentiment.yml"))}

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
    @llm_tasks = "spacy.Sentiment.v1"

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "sentiment.jinja2"))}

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
def test_sentiment_config(cfg_string, request):
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
def test_sentiment_predict(cfg_string, request):
    """Use OpenAI to get zero-shot sentiment results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    if cfg_string != "ext_template_cfg_string":
        assert nlp("This is horrible.")._.sentiment == 0
        assert 0 < nlp("This is meh.")._.sentiment <= 0.5
        assert nlp("This is perfect.")._.sentiment == 1


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize(
    "cfg_string_field",
    [
        ("zeroshot_cfg_string", None),
        ("fewshot_cfg_string", None),
        ("zeroshot_cfg_string", "sentiment_x"),
    ],
)
def test_lemma_io(cfg_string_field, request):
    cfg_string, field = cfg_string_field
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    if field:
        orig_config["components"]["llm"]["task"]["field"] = field
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    score = getattr(nlp2("This is perfect.")._, field if field else "sentiment")
    if cfg_string != "ext_template_cfg_string":
        assert score == 1


def test_jinja_template_rendering_without_examples():
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "They're indifferent."
    doc = nlp.make_doc(text)

    sentiment_task = make_sentiment_task(examples=None)
    prompt = list(sentiment_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == f"""
Analyse whether the text surrounded by ''' is positive or negative. Respond with a float value between 0 and 1. 1 represents an exclusively positive sentiment, 0 an exclusively negative sentiment.

Text:
'''
{text}
'''
Answer:""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "sentiment.json"),
        str(EXAMPLES_DIR / "sentiment.yml"),
        str(EXAMPLES_DIR / "sentiment.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "It was the happiest day of her life."
    doc = nlp.make_doc(text)

    sentiment_task = make_sentiment_task(examples=fewshot_reader(examples_path))
    prompt = list(sentiment_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
        == """
Analyse whether the text surrounded by ''' is positive or negative. Respond with a float value between 0 and 1. 1 represents an exclusively positive sentiment, 0 an exclusively negative sentiment.
Below are some examples (only use these as a guide):

Text:
'''
This is horrifying.
'''
Answer: 0.0

Text:
'''
This is underwhelming.
'''
Answer: 0.25

Text:
'''
This is ok.
'''
Answer: 0.5

Text:
'''
I'm looking forward to this!
'''
Answer: 1.0

Text:
'''
It was the happiest day of her life.
'''
Answer:""".strip()
    )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "sentiment.jinja2")
    template = file_reader(template_path)
    text = "There is a silver lining."
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(text)

    sentiment_task = make_sentiment_task(template=template)
    prompt = list(sentiment_task.generate_prompts([doc]))[0]
    assert (
        prompt.strip()
        == f"""
Text: {text}
Sentiment:
""".strip()
    )
