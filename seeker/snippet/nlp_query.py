#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

﻿import spacy
import pandas as pd
from .parse_nlp_query import parse_nlp_query

nlp = spacy.load("en_core_web_sm")

def handle_nlp_query(query, data):
    try:
        condition = parse_nlp_query(query)
        if not condition:
            print("Query resulted in no conditions. Please check your input.")
            return data
        result = data.query(condition)
        return result
    except pd.core.computation.ops.UndefinedVariableError as e:
        print(f"Error: Undefined variable in query. Did you mean to use a column name from: {', '.join(data.columns)}?")
    except pd.core.computation.ops.ParsingError as e:
        print(f"Syntax error in query. Common mistakes include: missing quotation marks around strings, incorrect comparison operators. Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Please check your query syntax.")
    return data

def feedback_on_query(query):
    condition = parse_nlp_query(query)
    if condition:
        print(f"Your query will be interpreted as: {condition}")
    else:
        print("Your query did not result in any conditions. Please verify your query.")