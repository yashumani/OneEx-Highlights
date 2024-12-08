import sys
import os

# Set an alternative PYTHONPATH
def set_custom_pythonpath(custom_path):
    """
    Add a custom PYTHONPATH to sys.path for module imports.
    
    Args:
    custom_path (str): The path to add to PYTHONPATH.
    """
    if custom_path not in sys.path:
        sys.path.insert(0, custom_path)

# Set your desired PYTHONPATH
custom_pythonpath = "/path/to/your/pythonpath"
set_custom_pythonpath(custom_pythonpath)

# Import required libraries (ensure they are available in the custom PYTHONPATH)
import pandas as pd
from transformers import pipeline

# Load the summarization model
def load_summarization_model(model_name):
    """
    Load a summarization model from Hugging Face transformers.
    """
    return pipeline("summarization", model=model_name)

# Function to summarize narratives
def summarize_text(text, summarizer, max_length=130, min_length=30):
    """
    Summarize a given text using a summarization model.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"

# Process data for summarization
def process_summarization(input_table, summarizer, narrative_column="Narrative", max_length=130, min_length=30):
    """
    Summarize narratives in the input table.
    """
    if narrative_column not in input_table.columns:
        raise ValueError(f"'{narrative_column}' column not found in the input table.")
    
    # Generate summaries
    input_table["Summary"] = input_table[narrative_column].apply(
        lambda x: summarize_text(x, summarizer, max_length=max_length, min_length=min_length) if pd.notna(x) else "No text to summarize"
    )
    return input_table

# Main function
def phase4(df):
    """
    Phase 4 function to summarize narratives in the input dataframe.
    
    Args:
    df (pd.DataFrame): Input data frame, expected to have a "Narrative" column.
    
    Returns:
    pd.DataFrame: Dataframe with added 'Summary' column.
    """
    # Model configuration
    model_name = "facebook/bart-large-cnn"  # You can switch to "t5-small" or "t5-large" if needed
    summarizer = load_summarization_model(model_name)
    
    # Narrative column configuration
    narrative_column = "Narrative"
    if narrative_column in df.columns:
        # Summarization parameters
        max_length = 130
        min_length = 30

        # Summarize narratives
        summarized_table = process_summarization(
            input_table=df,
            summarizer=summarizer,
            narrative_column=narrative_column,
            max_length=max_length,
            min_length=min_length
        )

        return summarized_table
    else:
        raise ValueError(f"Column '{narrative_column}' not found in the input data.")

# Example of how to set custom PYTHONPATH and use the function
if __name__ == "__main__":
    # Set your desired PYTHONPATH dynamically if needed
    custom_pythonpath = "C:/Users/Sharya3/AppData/Local/Programs/Python/Python312"
    set_custom_pythonpath(custom_pythonpath)
    
    df = input_table_1.copy()
	# Run Phase 4 processing
    result = phase4(df)
    print(result)
    output_table_1 = summarized_table
