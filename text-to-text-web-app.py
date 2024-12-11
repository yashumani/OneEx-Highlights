import streamlit as st
from transformers import pipeline
from language_tool_python import LanguageTool

# Initialize Summarization Models
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
t5_summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Initialize Grammar Correction Tool
grammar_tool = LanguageTool("en-US")

# Function for Summarization and Grammar Correction
def summarize_and_correct(text, model, max_length=150, min_length=50):
    if model == "BART":
        summarizer = bart_summarizer
    elif model == "T5":
        summarizer = t5_summarizer
    else:
        return "Invalid model selected."

    # Generate the summary
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        return f"Error in summarization: {e}"

    # Correct grammar in the summary
    corrected_summary = grammar_tool.correct(summary)
    return corrected_summary

# Streamlit UI
st.title("Text Summarization and Grammar Correction App")
st.markdown("Select a summarization model, input your text, and get a concise, grammatically correct summary!")

# User input for the text
text_input = st.text_area("Enter the text to summarize:", height=200)

# User input for model selection
model_choice = st.selectbox("Select Summarization Model", ["BART", "T5"])

# Parameters for summarization
max_length = st.slider("Maximum Summary Length", 50, 300, 150)
min_length = st.slider("Minimum Summary Length", 10, 100, 50)

# Button to trigger summarization
if st.button("Summarize and Correct"):
    if text_input.strip() == "":
        st.error("Please enter text to summarize.")
    else:
        st.info("Generating summary...")
        corrected_summary = summarize_and_correct(text_input, model_choice, max_length, min_length)
        st.success("Summary Generated!")
        st.subheader("Corrected Summary:")
        st.write(corrected_summary)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, Hugging Face Transformers, and LanguageTool.")
