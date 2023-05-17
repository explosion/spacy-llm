from spacy import util
from spacy_streamlit import visualize_ner, visualize_textcat
import streamlit as st

DEFAULT_TEXT = "Matt Honnibal is the CEO of Explosion."

st.title("spacy-llm Streamlit Demo")
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=70)

textcat_config = util.load_config(
    "usage_examples/streamlit/openai_textcat_zeroshot.cfg"
)
textcat_model = util.load_model_from_config(textcat_config, auto_fill=True)

ner_config = util.load_config("usage_examples/streamlit/openai_ner_zeroshot.cfg")
ner_model = util.load_model_from_config(ner_config, auto_fill=True)

models = {"textcat": textcat_model, "ner": ner_model}
model_names = models.keys()

selected_model = st.sidebar.selectbox("Model", model_names)

nlp = models[selected_model]
doc = nlp(text)
prompt = "\n".join([str(prompt) for prompt in nlp.get_pipe("llm")._template([doc])])
st.sidebar.markdown("## OpenAI Prompt:")
st.sidebar.text(prompt)

if selected_model == "textcat":
    visualize_textcat(doc)
if selected_model == "ner":
    visualize_ner(doc)
