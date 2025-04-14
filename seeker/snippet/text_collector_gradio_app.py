#date: 2025-04-14T16:43:12Z
#url: https://api.github.com/gists/8bc7a8b97b571e44b7dcf18eeb9976fb
#owner: https://api.github.com/users/patrickfleith

import gradio as gr
import os
import jsonlines
import random
from datetime import datetime

# Define the path for the output JSONL file
OUTPUT_FILE = "collected_texts.jsonl"

# List of success emojis for variety
SUCCESS_EMOJIS = ["✅", "🎉", "👍", "🚀", "💯", "⭐", "🌟", "🔥", "💪", "🏆"]

def count_rows():
    """
    Count the number of rows in the JSONL file.
    """
    if not os.path.exists(OUTPUT_FILE):
        return 0
    
    count = 0
    try:
        with jsonlines.open(OUTPUT_FILE, mode='r') as reader:
            for _ in reader:
                count += 1
    except Exception:
        pass
    
    return count

def save_text(text):
    """
    Save the submitted text to a JSONL file.
    Each entry is saved as a new line with a 'text' field.
    Returns the status message and the updated row count.
    """
    if not text.strip():
        return "Error: Empty text. Please enter some text to save.", count_rows(), ""
    
    try:
        # Ensure the file exists
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'w') as f:
                pass
        
        # Append the new text entry to the JSONL file
        with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
            writer.write({"text": text})
        
        # Get the current timestamp and a random emoji for the confirmation message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emoji = random.choice(SUCCESS_EMOJIS)
        
        # Return success message, updated count, and empty string to clear the text area
        return f"{emoji} Text saved successfully at {timestamp}!", count_rows(), ""
    
    except Exception as e:
        return f"Error saving text: {str(e)}", count_rows(), ""

# Create the Gradio interface
with gr.Blocks(title="Text Collector") as app:
    gr.Markdown("# Text Collector")
    gr.Markdown("Paste your text below and click 'Submit' to save it to the JSONL file.")
    
    # Initialize row counter
    row_count = count_rows()
    
    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Paste your text here...",
            lines=20,  # Increased text area size
            label="Text Input",
            autofocus=True
        )
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
    
    output = gr.Textbox(label="Status")
    counter = gr.Number(value=row_count, label="Total Entries", precision=0)
    
    # Set up the submission action
    submit_btn.click(
        fn=save_text,
        inputs=text_input,
        outputs=[output, counter, text_input]  # Added counter and text_input to clear it
    )
    
    # Display file information
    gr.Markdown(f"Data is being saved to: `{os.path.abspath(OUTPUT_FILE)}`")

if __name__ == "__main__":
    app.launch()
