import os
import pandas as pd
import streamlit as st
from transformers import pipeline
from pathlib import Path
import torch
from TTS.api import TTS  # Using Coqui TTS for Text-to-Speech

# Define Summarization Models
SUMMARIZATION_MODELS = {
    "BART": "facebook/bart-large-cnn",
    "T5-Small": "t5-small",
    "Pegasus": "google/pegasus-large",
}

# Load Summarization Model
def load_summarization_model(model_name):
    """
    Load a summarization model from Hugging Face transformers.
    """
    return pipeline("summarization", model=model_name)

# Summarize Text
def summarize_text(text, summarizer, max_length=130, min_length=30):
    """
    Summarize a given text using a summarization model.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"

# Process Summarization
def process_summarization(input_table, summarizer, narrative_column="Narrative", max_length=130, min_length=30):
    """
    Summarize narratives in the input table.
    """
    if narrative_column not in input_table.columns:
        raise ValueError(f"'{narrative_column}' column not found in the input table.")
    
    input_table["Summary"] = input_table[narrative_column].apply(
        lambda x: summarize_text(x, summarizer, max_length=max_length, min_length=min_length) if pd.notna(x) else "No text to summarize"
    )
    return input_table

# Load Text-to-Speech Model
def load_tts_model():
    """
    Load Coqui TTS model for speech synthesis.
    """
    return TTS(model_name="tts_models/en/ljspeech/vits")

# Generate Podcast
def generate_podcast(summaries, tts_model, output_path="output_podcast.wav"):
    """
    Generate a podcast audio file from summaries using TTS.
    """
    if not summaries:
        raise ValueError("No summaries provided for podcast generation.")
    
    # Combine all summaries into a single string
    podcast_text = " ".join(summaries)
    
    # Generate audio
    tts_model.tts_to_file(text=podcast_text, file_path=output_path)
    return output_path

# Streamlit App
def main():
    st.title("Summarization & Podcast Generation")
    st.markdown("Upload a CSV file containing narratives, and generate summaries or a podcast.")

    # File Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        # Load CSV
        input_table = pd.read_csv(uploaded_file)
        st.write("Input Table Preview:")
        st.dataframe(input_table.head())

        # Select Narrative Column
        narrative_column = st.selectbox("Select the column containing narratives", input_table.columns)

        # Select Summarization Model
        model_choice = st.selectbox("Choose a summarization model", list(SUMMARIZATION_MODELS.keys()))
        model_name = SUMMARIZATION_MODELS[model_choice]
        summarizer = load_summarization_model(model_name)

        # Summarization
        if st.button("Summarize Narratives"):
            summarized_table = process_summarization(input_table, summarizer, narrative_column=narrative_column)
            st.write("Summarized Table:")
            st.dataframe(summarized_table)

            # Download Summarized Table
            csv = summarized_table.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Summarized Table", data=csv, file_name="summarized_table.csv", mime="text/csv")

            # Text-to-Speech
            tts_model = load_tts_model()
            summaries = summarized_table["Summary"].tolist()

            if st.button("Generate Podcast"):
                podcast_path = generate_podcast(summaries, tts_model)
                st.success(f"Podcast generated: {podcast_path}")
                st.audio(podcast_path, format="audio/wav")

# Run Streamlit App
if __name__ == "__main__":
    main()
