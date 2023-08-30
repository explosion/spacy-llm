# ruff: noqa: W291
import csv
import functools
from pathlib import Path

import numpy
import pytest
import spacy
import srsly
from confection import Config
from spacy import Vocab
from spacy.kb import InMemoryLookupKB
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.registry import fewshot_reader, file_reader
from spacy_llm.tasks.entity_linker import EntityLinkerTask, make_entitylinker_task
from spacy_llm.util import assemble_from_config

from ...tasks.entity_linker.registry import make_candidate_selector_pipeline
from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@functools.lru_cache()
def build_el_pipeline(nlp_path: Path, desc_path: Path) -> None:
    """Builds and persists pipeline with untrained EL component and initialized toy knowledge base.
    nlp_path (Path): Path to store pipeline under.
    desc_path (Path): Path to store descriptions file under.
    """
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entity_linker")
    kb = InMemoryLookupKB(
        vocab=nlp.vocab, entity_vector_length=nlp.vocab.vectors_length
    )

    # Define entities.
    kb_data = dict(srsly.read_yaml(Path(__file__).parent / "misc" / "el_kb_data.yaml"))
    entities = kb_data["entities"]
    qids = list(entities.keys())

    # Set entities with dummy values for embeddings and frequencies.
    vec_shape = spacy.load("en_core_web_md")(" ").vector.shape
    kb.set_entities(
        entity_list=qids,
        vector_list=[numpy.zeros(vec_shape)] * len(qids),
        freq_list=[1] * len(qids),
    )

    # Add aliases and dummy prior probabilities.
    for alias_data in kb_data["aliases"]:
        kb.add_alias(**alias_data)

    # Set KB in pipeline, persist.
    def load_kb(vocab: Vocab) -> InMemoryLookupKB:
        return kb

    nlp.get_pipe("entity_linker").set_kb(load_kb)
    nlp.to_disk(nlp_path)

    # Write descriptions to file.
    with open(desc_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=";")
        for qid, ent_desc in entities.items():
            csv_writer.writerow([qid, ent_desc["desc"]])


@pytest.fixture
def noop_config():
    return """
    [paths]
    el_nlp = null
    el_desc = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.EntityLinker.v1"

    [components.llm.model]
    @llm_models = "test.NoOpModel.v1"

    [initialize]
    [initialize.components]
    [initialize.components.llm]

    [initialize.components.llm.candidate_selector]
    @llm_misc = "spacy.CandidateSelector.v1"
    nlp_path = ${paths.el_nlp}
    desc_path = ${paths.el_desc}
    """


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [paths]
    el_nlp = null
    el_desc = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.EntityLinker.v1"

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {"temperature": 0}

    [initialize]
    [initialize.components]
    [initialize.components.llm]

    [initialize.components.llm.candidate_selector]
    @llm_misc = "spacy.CandidateSelector.v1"
    nlp_path = ${paths.el_nlp}
    desc_path = ${paths.el_desc}
    """


@pytest.fixture
def fewshot_cfg_string():
    return f"""
    [paths]
    el_nlp = null
    el_desc = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.EntityLinker.v1"

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "entity_linker.yml"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}

    [initialize]
    [initialize.components]
    [initialize.components.llm]

    [initialize.components.llm.candidate_selector]
    @llm_misc = "spacy.CandidateSelector.v1"
    nlp_path = ${{paths.el_nlp}}
    desc_path = ${{paths.el_desc}}
    """


@pytest.fixture
def ext_template_cfg_string():
    """Simple zero-shot config with an external template"""

    return f"""
    [paths]
    el_nlp = null
    el_desc = null

    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]
    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.EntityLinker.v1"

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "entity_linker.jinja2"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}

    [initialize]
    [initialize.components]
    [initialize.components.llm]

    [initialize.components.llm.candidate_selector]
    @llm_misc = "spacy.CandidateSelector.v1"
    nlp_path = ${{paths.el_nlp}}
    desc_path = ${{paths.el_desc}}
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
def test_entity_linker_config(cfg_string, request, tmp_path):
    cfg_string = request.getfixturevalue(cfg_string)
    config = Config().from_str(
        cfg_string,
        overrides={
            "paths.el_nlp": str(tmp_path),
            "paths.el_desc": str(tmp_path / "desc.csv"),
        },
    )
    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    nlp = spacy.util.load_model_from_config(config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]

    # also test nlp config from a dict in add_pipe
    component_cfg = dict(config["components"]["llm"])
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
def test_entity_linker_predict(cfg_string, request, tmp_path):
    """Use OpenAI to get zero-shot LEMMA results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(
        cfg,
        overrides={
            "paths.el_nlp": str(tmp_path),
            "paths.el_desc": str(tmp_path / "desc.csv"),
        },
    )
    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    nlp.initialize(lambda: [])

    text = "Alice goes to Boston to see the Boston Celtics game."
    doc = nlp.make_doc(text)
    doc.ents = [
        Span(doc=doc, start=3, end=4, label="LOC"),  # Q100
        Span(doc=doc, start=7, end=9, label="ORG"),  # Q131371
    ]
    doc = nlp(doc)
    if cfg_string != "ext_template_cfg_string":
        assert len(doc.ents) == 2
        assert doc.ents[0].kb_id_ == "Q100"
        assert doc.ents[1].kb_id_ == "Q131371"


@pytest.mark.external
@pytest.mark.parametrize(
    "cfg_string",
    [
        "zeroshot_cfg_string",
        "fewshot_cfg_string",
    ],
)
def test_el_io(cfg_string, request, tmp_path):
    cfg = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(
        cfg,
        overrides={
            "paths.el_nlp": str(tmp_path),
            "paths.el_desc": str(tmp_path / "desc.csv"),
        },
    )
    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    nlp.initialize(lambda: [])

    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]
    nlp2.initialize(lambda: [])

    text = "Alice goes to Boston to see the Boston Celtics game."
    doc = nlp.make_doc(text)
    doc.ents = [
        Span(doc=doc, start=3, end=4, label="LOC"),  # Q100
        Span(doc=doc, start=7, end=9, label="ORG"),  # Q131371
    ]
    doc = nlp2(doc)
    if cfg_string != "ext_template_cfg_string":
        assert len(doc.ents) == 2
        assert doc.ents[0].kb_id_ == "Q100"
        assert doc.ents[1].kb_id_ == "Q131371"


def test_jinja_template_rendering_without_examples(tmp_path):
    """Test if jinja template renders as we expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "Alice goes to Boston to see the Boston Celtics game."
    doc = nlp.make_doc(text)
    doc.ents = [
        Span(doc=doc, start=3, end=4, label="LOC"),
        Span(doc=doc, start=7, end=9, label="ORG"),
    ]

    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    el_task = make_entitylinker_task(examples=None)
    el_task._candidate_selector = make_candidate_selector_pipeline(
        nlp_path=tmp_path, desc_path=tmp_path / "desc.csv"
    )
    prompt = list(el_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip().replace(" \n", "\n")
        == """
For each of the MENTIONS in the TEXT, resolve the MENTION to the correct entity listed in ENTITIES.
Each of the ENTITIES is prefixed by its ENTITY ID. Each of the MENTIONS in the TEXT is surrounded by *.
For each of the MENTIONS appearing in the text, output the ID of the description fitting them best.
This ID has to be surrounded by single <>, for example <1>.
Make sure you make a choice for each of the MENTIONS. Prefix the solution for each MENTION with "--- ".
Output the chosen solution immediately after "--- ".
For each MENTION, describe your reasoning process in a single sentence.

TEXT:
'''
Alice goes to *Boston* to see the *Boston Celtics* game.
'''
MENTIONS: *Boston*, *Boston Celtics*
ENTITIES:
- For *Boston*:
    Q100. city in and state capital of Massachusetts, United States
    Q131371. NBA team based in Boston; tied with most NBA Championships
    Q204289. American rock band
    Q311975. town in Lincolnshire, England
    Q671475. airport in Boston, Massachusetts, United States
- For *Boston Celtics*:
    Q131371. NBA team based in Boston; tied with most NBA Championships
    Q107723060. The 2021–22 Boston Celtics season was the 76th season of the franchise in the National Basketball Association (NBA). Following the Celtics' first-round exit to the Brooklyn Nets in five games from las
    Q3643001. NBA basketball team season
    Q3466394. season of National Basketball Association team the Boston Celtics
    Q3642995. NBA basketball team season
SOLUTION:
""".strip().replace(
            " \n", "\n"
        )
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "entity_linker.json"),
        str(EXAMPLES_DIR / "entity_linker.yml"),
        str(EXAMPLES_DIR / "entity_linker.jsonl"),
    ],
)
def test_jinja_template_rendering_with_examples(examples_path, tmp_path):
    """Test if jinja2 template renders as expected

    We apply the .strip() method for each prompt so that we don't have to deal
    with annoying newlines and spaces at the edge of the text.
    """
    nlp = spacy.blank("xx")
    text = "Alice goes to Boston to see the Boston Celtics game."
    doc = nlp.make_doc(text)
    doc.ents = [
        Span(doc=doc, start=3, end=4, label="LOC"),
        Span(doc=doc, start=7, end=9, label="ORG"),
    ]

    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    el_task = make_entitylinker_task(examples=fewshot_reader(examples_path))
    el_task._candidate_selector = make_candidate_selector_pipeline(
        nlp_path=tmp_path, desc_path=tmp_path / "desc.csv"
    )
    prompt = list(el_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip().replace(" \n", "\n")
        == """
For each of the MENTIONS in the TEXT, resolve the MENTION to the correct entity listed in ENTITIES.
Each of the ENTITIES is prefixed by its ENTITY ID. Each of the MENTIONS in the TEXT is surrounded by *.
For each of the MENTIONS appearing in the text, output the ID of the description fitting them best.
This ID has to be surrounded by single <>, for example <1>.
Make sure you make a choice for each of the MENTIONS. Prefix the solution for each MENTION with "--- ".
Output the chosen solution immediately after "--- ".
For each MENTION, describe your reasoning process in a single sentence.

Below are some examples (only use these as a guide):

TEXT:
'''
Alice goes to *New York* to see the *New York Knicks* game.
'''
MENTIONS:
ENTITIES:
- For *New York*:
    Q60. most populous city in the United States
    Q1384. U.S. state
- For *New York Knicks*:
    Q60. most populous city in the United States
    Q131364. National Basketball Association team in New York City
SOLUTION:
--- <Q60>
--- <Q131364>

TEXT:
'''
*New York* is called the *Big Apple*. It also has *Apple* stores.
'''
MENTIONS:
ENTITIES:
- For *New York*:
    Q60. most populous city in the United States
    Q1384. U.S. state
- For *Big Apple*:
    Q14435. nickname for New York City
    Q89. fruit of the apple tree
- For *Apple*:
    Q89. fruit of the apple tree
    Q312. American multinational technology company
SOLUTION:
--- <Q60> The mention of "Big Apple" in the same context clarifies that this is about the city New York.
--- <Q14435> Big Apple is a well-known nickname of New York.
--- <Q312> The context of "stores" indicates that this is about the technology company Apple, which operates "Apple stores".


End of examples.TEXT:
'''
Alice goes to *Boston* to see the *Boston Celtics* game.
'''
MENTIONS: *Boston*, *Boston Celtics*
ENTITIES:
- For *Boston*:
    Q100. city in and state capital of Massachusetts, United States
    Q131371. NBA team based in Boston; tied with most NBA Championships
    Q204289. American rock band
    Q311975. town in Lincolnshire, England
    Q671475. airport in Boston, Massachusetts, United States
- For *Boston Celtics*:
    Q131371. NBA team based in Boston; tied with most NBA Championships
    Q107723060. The 2021–22 Boston Celtics season was the 76th season of the franchise in the National Basketball Association (NBA). Following the Celtics' first-round exit to the Brooklyn Nets in five games from las
    Q3643001. NBA basketball team season
    Q3466394. season of National Basketball Association team the Boston Celtics
    Q3642995. NBA basketball team season
SOLUTION:
""".strip().replace(
            " \n", "\n"
        )
    )


def test_external_template_actually_loads(tmp_path):
    template_path = str(TEMPLATES_DIR / "entity_linker.jinja2")
    template = file_reader(template_path)
    text = "Alice and Bob went to the supermarket"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(text)

    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    el_task = make_entitylinker_task(template=template, examples=None)
    el_task._candidate_selector = make_candidate_selector_pipeline(
        nlp_path=tmp_path, desc_path=tmp_path / "desc.csv"
    )

    assert (
        list(el_task.generate_prompts([doc]))[0].strip()
        == f"""
This is a test entity linking template.
Here is the text: {text}
""".strip()
    )


@pytest.mark.parametrize("n_prompt_examples", [-1, 0, 1, 2])
def test_el_init(noop_config, n_prompt_examples: int, tmp_path):
    config = Config().from_str(
        noop_config,
        overrides={
            "paths.el_nlp": str(tmp_path),
            "paths.el_desc": str(tmp_path / "desc.csv"),
        },
    )
    build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    nlp = assemble_from_config(config)

    examples = []

    text = "Alice goes to Boston to see the Boston Celtics game."
    pred_1 = nlp.make_doc(text)
    pred_1.ents = [
        Span(doc=pred_1, start=3, end=4, label="LOC"),
        Span(doc=pred_1, start=7, end=9, label="ORG"),
    ]
    gold_1 = nlp.make_doc(text)
    gold_1.ents = [
        Span(doc=gold_1, start=3, end=4, label="LOC", kb_id="Q100"),
        Span(doc=gold_1, start=7, end=9, label="ORG", kb_id="Q131371"),
    ]
    examples.append(Example(pred_1, gold_1))

    text = "Alice goes to New York to see the New York Knicks game."
    pred_2 = nlp.make_doc(text)
    pred_2.ents = [
        Span(doc=pred_2, start=3, end=4, label="LOC"),
        Span(doc=pred_2, start=7, end=10, label="ORG"),
    ]
    gold_2 = nlp.make_doc(text)
    gold_2.ents = [
        Span(doc=gold_2, start=3, end=4, label="LOC", kb_id="Q60"),
        Span(doc=gold_2, start=7, end=10, label="ORG", kb_id="Q131364"),
    ]
    examples.append(Example(pred_2, gold_2))

    _, llm = nlp.pipeline[0]
    task: EntityLinkerTask = llm._task

    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        **nlp.config["initialize"]["components"]["llm"],
        "n_prompt_examples": n_prompt_examples,
    }
    nlp.initialize(lambda: examples)

    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(examples)


def test_ent_highlighting():
    """Tests highlighting of entities in text."""
    nlp = spacy.blank("xx")
    text = "Alice goes to Boston to see the Boston Celtics game."
    doc = nlp.make_doc(text)
    doc.ents = [
        Span(doc=doc, start=3, end=4, label="LOC"),
        Span(doc=doc, start=7, end=9, label="ORG"),
    ]

    assert (
        EntityLinkerTask.highlight_ents_in_text(doc)
        == "Alice goes to *Boston* to see the *Boston Celtics* game."
    )
