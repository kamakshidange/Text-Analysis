# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline

# --- Configuration ---
st.set_page_config(
    page_title="Hugging Face Transformer Demos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Pipelines with Caching ---
@st.cache_resource
def load_pipelines():
    """Load all necessary NLP pipelines once."""
    try:
        with st.spinner("Loading NLP models... This might take a moment!"):
            pipelines = {
                "text_gen": pipeline("text-generation", model="gpt2"),
                "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
                "sentiment": pipeline("sentiment-analysis"),
                "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
                "grammar": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
            }
        return pipelines
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure PyTorch and transformers are installed.")
        return None


# Load pipelines
pipes = load_pipelines()

# --- Main UI ---
st.title("🧠 NLP Tasks with Hugging Face Transformers")
st.markdown("Perform Text Generation, Summarization, Sentiment Analysis, NER and Grammar Correction.")

# Sidebar
task = st.sidebar.selectbox(
    "*Select an NLP Task*",
    [
        "Text Generation",
        "Summarization",
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Grammar Correction"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("✔ Models load only once using @st.cache_resource")

# Stop if pipelines didn't load
if not pipes:
    st.stop()


# ----------------------------------------------------------------------
# 1. TEXT GENERATION
# ----------------------------------------------------------------------
if task == "Text Generation":
    st.header("📝 Text Generation (GPT-2)")
    input_text = st.text_area("Enter your prompt:", "The future of AI is", height=120)
    max_len = st.slider("Max Output Length", 50, 500, 200)

    if st.button("Generate"):
        with st.spinner("Generating text..."):
            result = pipes["text_gen"](input_text, max_length=max_len, num_return_sequences=1)
        st.success("Done!")
        st.code(result[0]["generated_text"], language="text")


# ----------------------------------------------------------------------
# 2. SUMMARIZATION
# ----------------------------------------------------------------------
elif task == "Summarization":
    st.header("📑 Text Summarization (BART-Large-CNN)")
    input_text = st.text_area(
        "Enter long text:",
        height=300,
        value="""
        The Amazon rainforest is the largest tropical rainforest in the world.
        It covers over 5.5 million square kilometers and is primarily located
        within nine countries, with the majority in Brazil. The Amazon is
        crucial for the global climate and harbors immense biodiversity. 
        """
    )

    max_len = st.slider("Max Summary Length", 20, 200, 60)
    min_len = st.slider("Min Summary Length", 10, max_len - 10, 20)

    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = pipes["summarizer"](input_text, max_length=max_len, min_length=min_len)
        st.success("Summary Ready!")
        st.write(summary[0]["summary_text"])


# ----------------------------------------------------------------------
# 3. SENTIMENT ANALYSIS
# ----------------------------------------------------------------------
elif task == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    input_text = st.text_area("Enter text:", "The food was amazing!", height=120)

    if st.button("Analyze"):
        with st.spinner("Analyzing sentiment..."):
            result = pipes["sentiment"](input_text)[0]

        st.subheader("Result:")
        if result["label"] == "POSITIVE":
            st.success(f"POSITIVE (Confidence: {result['score']:.4f})")
        else:
            st.error(f"NEGATIVE (Confidence: {result['score']:.4f})")


# ----------------------------------------------------------------------
# 4. NER
# ----------------------------------------------------------------------
elif task == "Named Entity Recognition (NER)":
    st.header("📍 Named Entity Recognition")
    input_text = st.text_area(
        "Enter text:",
        "Elon Musk founded SpaceX in California.",
        height=120
    )

    if st.button("Identify Entities"):
        with st.spinner("Detecting entities..."):
            entities = pipes["ner"](input_text)

        st.success("Entities Identified:")
        st.dataframe([
            {"Entity": ent["word"], "Label": ent["entity_group"], "Score": f"{ent['score']:.4f}"}
            for ent in entities
        ])


# ----------------------------------------------------------------------
# 5. GRAMMAR CORRECTION
# ----------------------------------------------------------------------
elif task == "Grammar Correction":
    st.header("✍ Grammar Correction")
    input_text = st.text_area(
        "Enter incorrect text:",
        "She go to school every days, but he not going tomorrow.",
        height=120
    )

    if st.button("Correct"):
        with st.spinner("Fixing grammar..."):
            result = pipes["grammar"](input_text)[0]["generated_text"]

        st.success("Correction Complete!")
        st.markdown(f"**Original:** {input_text}")
        st.markdown(f"**Corrected:** {result}")
