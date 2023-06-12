import os
from typing import cast

import streamlit as st
from spacy.util import load_config_from_str
from spacy_streamlit import visualize_ner, visualize_textcat

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.util import assemble_from_config

NER_CONFIG = """
[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm]
factory = "llm"

[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "gpt-3.5-turbo", "temperature": 0.3}

[components.llm.task]
@llm_tasks = "spacy.NER.v2"
labels = PERSON,ORGANISATION,LOCATION
examples = null

[components.llm.task.normalizer]
@misc = "spacy.LowercaseNormalizer.v1"
"""

TEXTCAT_CONFIG = """
[nlp]
lang = "en"
pipeline = ["llm"]
batch_size = 128

[components]

[components.llm]
factory = "llm"

[components.llm.backend]
@llm_backends = "spacy.REST.v1"
api = "OpenAI"
config = {"model": "gpt-3.5-turbo", "temperature": 0.3}

[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = COMPLIMENT,INSULT
examples = null
exclusive_classes = true

[components.llm.task.normalizer]
@misc = "spacy.LowercaseNormalizer.v1"
"""

DEFAULT_TEXT = "Ernest Hemingway, born in Illinois, is generally considered one of the best authors of his time."

st.title("spacy-llm Streamlit Demo")
st.markdown(
    """
    The [spacy-llm](https://github.com/explosion/spacy-llm) package integrates
    Large Language Models (LLMs) into spaCy, featuring a modular system
    for fast prototyping and prompting, and turning unstructured responses
    into robust outputs for various NLP tasks, no training data required.

    This demo uses the OpenAI backend to demonstrate the NER and textcat
    tasks.
    """
)

os.environ["OPENAI_API_KEY"] = st.text_input(
    "Your OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", "")
)
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=70)

if os.environ["OPENAI_API_KEY"]:
    textcat_config = load_config_from_str(TEXTCAT_CONFIG)
    textcat_model = assemble_from_config(textcat_config)
    ner_config = load_config_from_str(NER_CONFIG)
    ner_model = assemble_from_config(ner_config)

    models = {"textcat": textcat_model, "ner": ner_model}
    model_names = models.keys()

    selected_model = st.sidebar.selectbox("Model", model_names)
    assert selected_model is not None

    nlp = models[selected_model]
    doc = nlp(text)
    llm_pipe = cast(LLMWrapper, nlp.get_pipe("llm"))
    prompt = "\n".join(
        [str(prompt) for prompt in llm_pipe._task.generate_prompts([doc])]
    )

    if selected_model == "textcat":
        visualize_textcat(doc)
    if selected_model == "ner":
        visualize_ner(doc)

    st.markdown("### Prompt:")
    st.text(prompt)
else:
    st.error("Input your OpenAI API key")
