#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import spacy
from spacy.training import Example
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the model path from environment variables
model_path = os.getenv("MODEL_PATH", "models/nlp_model")

# Sample training data
TRAIN_DATA = [
    ("Load data from source file.csv into my_table with options", {
        "entities": [(20, 29, "FILE_PATH"), (35, 43, "TABLE_NAME"), (50, 57, "OPTIONS")]
    }),
    ("Import users from data/users.json to user_list with date filter", {
        "entities": [(19, 34, "FILE_PATH"), (38, 47, "TABLE_NAME"), (53, 64, "OPTIONS")]
    }),
    # Add more examples to cover different variations
]

# Load an existing model
nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

# Add new labels
for label in ["FILE_TYPE", "FILE_PATH", "TABLE_NAME", "OPTIONS"]:
    ner.add_label(label)

# Training the model
optimizer = nlp.begin_training()
for itn in range(100):  # Adjust iterations as needed
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(losses)

# Save the trained model to disk
nlp.to_disk(model_path)  # Save to the path specified in the environment variable

# Example evaluation
test_text = "Import users from data/users.json to user_list with date filter"
doc = nlp(test_text)
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")