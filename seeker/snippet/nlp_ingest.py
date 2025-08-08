#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

﻿import spacy
from src.nlp.nlp_ingest.parse_nlp_ingest import parse_nlp_ingest
import pandas as pd

# Load the trained model
nlp = spacy.load("en_core_web_sm")

def load_data_by_type(file_type, file_path):
    # This function would load data based on file_type and file_path
    # For demonstration, we're just returning a dummy DataFrame
    if file_type == "users":
        return pd.DataFrame({'id': [1, 2], 'date_joined': ['2023-01-01', '2023-02-01'], 'file_path': file_path})
    elif file_type == "sales":
        return pd.DataFrame({'id': [1, 2], 'transaction_date': ['2023-01-01', '2023-02-01'], 'file_path': file_path})
    else:
        return pd.DataFrame()

def apply_filters(data, conditions):
    for condition in conditions:
        if 'options' in condition:
            # Here you would parse and apply the options. This is a placeholder.
            print(f"Applying options: {condition['options']}")
        if 'date_joined' in data.columns and 'date_joined' in condition:
            data = data[data['date_joined'] > condition['date_joined']]
    return data

def parse_nlp_ingest(query):
    doc = nlp(query)
    
    file_type = None
    file_path = None
    table_name = None
    options = None
    
    # Using the parsed conditions from parse_nlp_ingest.py
    conditions = parse_nlp_ingest(query)
    
    for condition in conditions:
        if 'file_type' in condition:
            file_type = condition['file_type']
        if 'file_path' in condition:
            file_path = condition['file_path']
        if 'table_name' in condition:
            table_name = condition['table_name']
        if 'options' in condition:
            options = condition['options']
    
    # Load data based on file type and path
    data = load_data_by_type(file_type, file_path)
    
    # Apply any filters or options
    if data is not None:
        data = apply_filters(data, conditions)
    
    if data.empty:
        return "Unable to determine data type to ingest or no data found after filtering."
    else:
        return data

def handle_nlp_ingest(query):
    return parse_nlp_ingest(query)