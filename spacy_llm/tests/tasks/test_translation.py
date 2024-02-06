from pathlib import Path

import pytest
import spacy
from confection import Config
from spacy.util import make_tempdir

from spacy_llm.registry import fewshot_reader, file_reader

from ...tasks import make_sentiment_task, make_translation_task
from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.Translation.v1"
    source_lang = "English"
    target_lang = "Spanish"

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
    @llm_tasks = "spacy.Translation.v1"
    source_lang = "English"
    target_lang = "Spanish"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "translation.yml"))}

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
    @llm_tasks = "spacy.Translation.v1"
    source_lang = "English"
    target_lang = "Spanish"

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "translation.jinja2"))}

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
def test_translation_config(cfg_string, request):
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
def test_translate_predict(cfg_string, request):
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg)
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    doc = nlp("This is the sun")
    if cfg_string != "ext_template_cfg_string":
        assert doc._.translation.strip(".") == "Este es el sol"


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
def test_translation_io(cfg_string_field, request):
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
    translation = getattr(nlp2("This is perfect.")._, field if field else "translation")
    if cfg_string != "ext_template_cfg_string":
        assert translation == "Esto es perfecto."


@pytest.mark.parametrize("source_lang", [None, "English"])
def test_jinja_template_rendering_without_examples(source_lang: str):
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("en")
    text = "They're indifferent."
    doc = nlp.make_doc(text)

    translation_task = (
        make_translation_task(
            examples=None, target_lang="Spanish", source_lang=source_lang
        )
        if source_lang
        else make_translation_task(examples=None, target_lang="Spanish")
    )
    prompt = list(translation_task.generate_prompts([doc]))[0][0][0]

    if source_lang:
        assert (
            prompt.strip()
            == f"""
Translate the text after "Text:" from English to Spanish.

Respond after "Translation:" with nothing but the translated text.

Text:
{text}
Translation:""".strip()
        )
    else:
        assert (
            prompt.strip()
            == f"""
Translate the text after "Text:" to Spanish.

Respond after "Translation:" with nothing but the translated text.

Text:
{text}
Translation:""".strip()
        )


@pytest.mark.parametrize("source_lang", [None, "English"])
@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "translation.json"),
        str(EXAMPLES_DIR / "translation.yml"),
        str(EXAMPLES_DIR / "translation.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path, source_lang):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("en")
    text = "It was the happiest day of her life."
    doc = nlp.make_doc(text)

    examples = fewshot_reader(examples_path)
    translation_task = (
        make_translation_task(
            examples=examples, target_lang="Spanish", source_lang=source_lang
        )
        if source_lang
        else make_translation_task(examples=examples, target_lang="Spanish")
    )
    prompt = list(translation_task.generate_prompts([doc]))[0][0][0]

    if source_lang:
        assert (
            prompt.strip()
            == """
Translate the text after "Text:" from English to Spanish.

Respond after "Translation:" with nothing but the translated text.
Below are some examples (only use these as a guide):

Text:
Top of the morning to you!
Translation:
¡Muy buenos días!

Text:
The weather is great today.
Translation:
El clima está fantástico hoy.

Text:
Do you know what will happen tomorrow?
Translation:
¿Sabes qué pasará mañana?

Text:
It was the happiest day of her life.
Translation:""".strip()
        )

    else:
        assert (
            prompt.strip()
            == """
Translate the text after "Text:" to Spanish.

Respond after "Translation:" with nothing but the translated text.
Below are some examples (only use these as a guide):

Text:
Top of the morning to you!
Translation:
¡Muy buenos días!

Text:
The weather is great today.
Translation:
El clima está fantástico hoy.

Text:
Do you know what will happen tomorrow?
Translation:
¿Sabes qué pasará mañana?

Text:
It was the happiest day of her life.
Translation:""".strip()
        )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "translation.jinja2")
    template = file_reader(template_path)
    text = "There is a silver lining."
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)

    sentiment_task = make_sentiment_task(template=template)
    prompt = list(sentiment_task.generate_prompts([doc]))[0][0][0]
    assert (
        prompt.strip()
        == f"""
Text: {text}
Translation:""".strip()
    )
