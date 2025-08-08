#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

﻿import spacy
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the model path from environment variables
model_path = os.getenv("MODEL_PATH", "models/nlp_model")

# Load the trained model
nlp = spacy.load(model_path)

def parse_nlp_ingest(query):
    doc = nlp(query)
    
    conditions = []
    
    for ent in doc.ents:
        if ent.label_ == "FILE_TYPE":
            conditions.append({"file_type": ent.text})
        elif ent.label_ == "FILE_PATH":
            conditions.append({"file_path": ent.text})
        elif ent.label_ == "TABLE_NAME":
            conditions.append({"table_name": ent.text})
        elif ent.label_ == "OPTIONS":
            conditions.append({"options": ent.text})
    
    return conditions

# Example usage
if __name__ == "__main__":
    query = "Load data from source file.csv into my_table with options"
    print(parse_nlp_ingest(query))