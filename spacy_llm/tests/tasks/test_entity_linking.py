# ruff: noqa: W291
import csv
import functools
from pathlib import Path

import numpy
import pytest
import spacy
from confection import Config
from spacy import Language, Vocab
from spacy.kb import InMemoryLookupKB
from spacy.tokens import Span
from spacy.training import Example
from spacy.util import make_tempdir

from spacy_llm.registry import fewshot_reader, file_reader
from spacy_llm.tasks.entity_linking import EntityLinkingTask
from spacy_llm.tasks.entity_linking import SpaCyPipelineCandidateSelector
from spacy_llm.tasks.entity_linking import make_entitylinking_task
from spacy_llm.util import assemble_from_config

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@functools.lru_cache()
def _build_el_pipeline(nlp_path: Path, desc_path: Path) -> Language:
    """Builds and persists pipeline with untrained EL component and initialized toy knowledge base.
    nlp_path (Path): Path to store pipeline under.
    descriptions_path (Path): Path to store descriptions file under.
    """
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entity_linker")
    kb = InMemoryLookupKB(
        vocab=nlp.vocab, entity_vector_length=nlp.vocab.vectors_length
    )

    # Define entities (move to file?).
    entities = {
        "Q100": {
            "name": "Boston",
            "desc": "city in and state capital of Massachusetts, United States",
        },
        "Q131371": {
            "name": "Boston Celtics",
            "desc": "NBA team based in Boston; tied with most NBA Championships",
        },
        "Q204289": {"name": "Boston", "desc": "American rock band"},
        "Q311975": {"name": "Boston", "desc": "town in Lincolnshire, England"},
        "Q671475": {
            "name": "Logan International Airport",
            "desc": "airport in Boston, Massachusetts, United States",
        },
        "Q107723060": {
            "name": "2021–22 Boston Celtics season",
            "desc": "The 2021–22 Boston Celtics season was the 76th season of the franchise in the National Basketball"
            " Association (NBA). Following the Celtics' first-round exit to the Brooklyn Nets in five games"
            " from las",
        },
        "Q3643001": {"name": "Boston", "desc": "NBA basketball team season"},
        "Q3466394": {
            "name": "Boston",
            "desc": "season of National Basketball Association team the Boston Celtics",
        },
        "Q3642995": {"name": "Boston", "desc": "NBA basketball team season"},
        "Q60": {"name": "New York", "desc": "most populous city in the United States"},
        "Q1384": {"name": "New York", "desc": "U.S. state"},
        "Q131364": {
            "name": "New York Knicks",
            "desc": "National Basketball Association team in New York City",
        },
        "Q14435": {"name": "Big Apple", "desc": "nickname for New York City"},
        "Q89": {"name": "Apple", "desc": "fruit of the apple tree"},
        "Q312": {"name": "Apple", "desc": "American multinational technology company"},
    }
    qids = list(entities.keys())

    # Set entities with dummy values for embeddings and frequencies.
    vec_shape = spacy.load("en_core_web_md")(" ").vector.shape
    kb.set_entities(
        entity_list=qids,
        vector_list=[numpy.zeros(vec_shape)] * len(qids),
        freq_list=[1] * len(qids),
    )

    # Add aliases and dummy prior probabilities.
    kb.add_alias(
        alias="Boston",
        entities=["Q100", "Q131371", "Q204289", "Q311975", "Q671475"],
        probabilities=[0.5, 0.2, 0.12, 0.1, 0.08],
    )
    kb.add_alias(
        alias="Boston Celtics",
        entities=["Q131371", "Q107723060", "Q3643001", "Q3466394", "Q3642995"],
        probabilities=[0.5, 0.2, 0.12, 0.1, 0.08],
    )
    kb.add_alias(
        alias="New York",
        entities=["Q60", "Q1384"],
        probabilities=[0.6, 0.4],
    )
    kb.add_alias(
        alias="New York Knicks",
        entities=["Q60", "Q131364"],
        probabilities=[0.6, 0.4],
    )
    kb.add_alias(
        alias="Big Apple",
        entities=["Q14435", "Q89"],
        probabilities=[0.6, 0.4],
    )
    kb.add_alias(
        alias="Apple",
        entities=["Q89", "Q312"],
        probabilities=[0.6, 0.4],
    )

    # Set KB in pipeline, persist.
    def load_kb(vocab: Vocab) -> InMemoryLookupKB:
        return kb

    nlp.get_pipe("entity_linker").set_kb(load_kb)
    nlp.to_disk(nlp_path)

    # Write descriptions to file.
    with open(desc_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for qid, ent_desc in entities.items():
            csv_writer.writerow([qid, ent_desc["desc"]])


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
    @llm_tasks = "spacy.EntityLinking.v1"

    [components.llm.task.candidate_selector]
    @llm_misc = "spacy.CandidateSelectorPipeline.v1"
    nlp_path = TO_REPLACE
    desc_path = TO_REPLACE

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
    @llm_tasks = "spacy.EntityLinking.v1"

    [components.llm.task.candidate_selector]
    @llm_misc = "spacy.CandidateSelectorPipeline.v1"
    nlp_path = TO_REPLACE
    desc_path = TO_REPLACE

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
    @llm_tasks = "spacy.EntityLinking.v1"

    [components.llm.task.candidate_selector]
    @llm_misc = "spacy.CandidateSelectorPipeline.v1"
    nlp_path = TO_REPLACE
    desc_path = TO_REPLACE

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str((Path(__file__).parent / "examples" / "entity_linking.yml"))}

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
    @llm_tasks = "spacy.EntityLinking.v1"

    [components.llm.task.candidate_selector]
    @llm_misc = "spacy.CandidateSelectorPipeline.v1"
    nlp_path = TO_REPLACE
    desc_path = TO_REPLACE

    [components.llm.task.template]
    @misc = "spacy.FileReader.v1"
    path = {str((Path(__file__).parent / "templates" / "entity_linking.jinja2"))}

    [components.llm.model]
    @llm_models = "spacy.GPT-3-5.v1"
    config = {{"temperature": 0}}
    """


def _update_cand_selector_paths_in_config(config: Config, tmp_path: Path) -> Config:
    """Updates paths for candidate selector in config and builds EL pipeline with KB with correspondig paths.
    config (Dict[str, Any]): Config to update.
    tmp_path (Path): Base directory for pipeline and descriptions file.
    RETURNS (Dict[str, Any]): Update config.
    """
    config["components"]["llm"]["task"]["candidate_selector"]["nlp_path"] = str(
        tmp_path
    )
    config["components"]["llm"]["task"]["candidate_selector"]["desc_path"] = str(
        tmp_path / "desc.csv"
    )
    _build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")

    return config


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
def test_entity_linking_config(cfg_string, request, tmp_path):
    cfg_string = request.getfixturevalue(cfg_string)
    config = _update_cand_selector_paths_in_config(
        Config().from_str(cfg_string), tmp_path
    )
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
def test_entity_linking_predict(cfg_string, request, tmp_path):
    """Use OpenAI to get zero-shot LEMMA results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg = request.getfixturevalue(cfg_string)
    orig_config = _update_cand_selector_paths_in_config(
        Config().from_str(cfg), tmp_path
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
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
    orig_config = _update_cand_selector_paths_in_config(
        Config().from_str(cfg), tmp_path
    )
    nlp = spacy.util.load_model_from_config(orig_config, auto_fill=True)
    assert nlp.pipe_names == ["llm"]
    # ensure you can save a pipeline to disk and run it after loading
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        nlp2 = spacy.load(tmpdir)
    assert nlp2.pipe_names == ["llm"]

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

    _build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    el_task = make_entitylinking_task(
        examples=None,
        candidate_selector=SpaCyPipelineCandidateSelector(
            nlp_path=tmp_path, desc_path=tmp_path / "desc.csv"
        ),
    )
    prompt = list(el_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
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
""".strip()
    )


@pytest.mark.parametrize(
    "examples_path",
    [
        str(EXAMPLES_DIR / "entity_linking.json"),
        str(EXAMPLES_DIR / "entity_linking.yml"),
        str(EXAMPLES_DIR / "entity_linking.jsonl"),
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

    _build_el_pipeline(nlp_path=tmp_path, desc_path=tmp_path / "desc.csv")
    el_task = make_entitylinking_task(
        examples=fewshot_reader(examples_path),
        candidate_selector=SpaCyPipelineCandidateSelector(
            nlp_path=tmp_path, desc_path=tmp_path / "desc.csv"
        ),
    )
    prompt = list(el_task.generate_prompts([doc]))[0]

    assert (
        prompt.strip()
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
--- <Q312> The context of "stores" indicates that this is about the technology company Apple, who operates "Apple stores".


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
""".strip()
    )


def test_external_template_actually_loads():
    template_path = str(TEMPLATES_DIR / "entity_linking.jinja2")
    template = file_reader(template_path)
    text = "Alice and Bob went to the supermarket"
    nlp = spacy.blank("xx")
    doc = nlp.make_doc(text)

    el_task = make_entitylinking_task(
        template=template,
        examples=None,
        candidate_selector=SpaCyPipelineCandidateSelector(
            nlp_path="/home/raphael/dev/spacy-projects/benchmarks/nel/training/mewsli_9/cg-default/model-best",
            desc_path="/home/raphael/dev/wikid/output/en/descriptions.csv",
        ),
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
    config = _update_cand_selector_paths_in_config(
        Config().from_str(noop_config), tmp_path
    )
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
    task: EntityLinkingTask = llm._task

    assert not task._prompt_examples

    nlp.config["initialize"]["components"]["llm"] = {
        "n_prompt_examples": n_prompt_examples
    }
    nlp.initialize(lambda: examples)

    if n_prompt_examples >= 0:
        assert len(task._prompt_examples) == n_prompt_examples
    else:
        assert len(task._prompt_examples) == len(examples)
