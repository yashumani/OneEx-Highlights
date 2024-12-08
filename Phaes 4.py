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
def main():
    # Model configuration
    model_name = "facebook/bart-large-cnn"  # You can switch to "t5-small" or "t5-large" if needed
    summarizer = load_summarization_model(model_name)
    
    # Load input_table_1 (output_table_1 from Phase 3)
    input_file = "output_table_1.csv"  # Replace with the correct path to your Phase 3 output
    input_table_1 = pd.read_csv(input_file)

    # Narrative column configuration
    narrative_column = "Narrative"
    if narrative_column in input_table_1.columns:
        # Summarization parameters
        max_length = 130
        min_length = 30

        # Summarize narratives
        summarized_table = process_summarization(
            input_table=input_table_1,
            summarizer=summarizer,
            narrative_column=narrative_column,
            max_length=max_length,
            min_length=min_length
        )

        # Assign summarized table to output_table_1
        global output_table_1
        output_table_1 = summarized_table

        # Print the summarized table for verification
        print("Summarized Data:")
        print(output_table_1.head())
    else:
        raise ValueError(f"Column '{narrative_column}' not found in the input data.")

if __name__ == "__main__":
    main()
