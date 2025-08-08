#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import pandas as pd
import json
from chardet import detect
from datetime import datetime
import sqlparse
from src.nlp.nlp_query.nlp_query import handle_nlp_query
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_encoding_and_query(filename, condition):
    try:
        if filename.endswith('.json'):
            with open(filename, 'rb') as file:
                raw_data = file.read()
                result = detect(raw_data)
                detected_encoding = result['encoding']
                confidence = result['confidence']
                logging.info(f"Detected encoding: {detected_encoding} with confidence {confidence}")

            if confidence > 0.7:
                df = load_data(filename, detected_encoding)
            else:
                df = attempt_common_encodings(filename)
        else:
            with open(filename, 'rb') as file:
                raw_data = file.read()
                result = detect(raw_data)
                detected_encoding = result['encoding']
                confidence = result['confidence']
                logging.info(f"Detected encoding: {detected_encoding} with confidence {confidence}")

            if confidence > 0.7:
                df = load_data(filename, detected_encoding)
            else:
                df = attempt_common_encodings(filename)
        
        if condition:
            if condition.strip().lower().startswith(('select', 'with', 'insert', 'update', 'delete')):
                return apply_query(df, condition)
            else:
                return handle_nlp_query(condition, df)
        else:
            return df
    except Exception as e:
        logging.error(f"Error querying data: {str(e)}")
        return None

def load_data(filename, encoding):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, encoding=encoding)
        df = convert_date_columns(df)
    elif filename.endswith('.json'):
        with open(filename, 'r', encoding=encoding) as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df = convert_date_columns(df)
    else:
        raise ValueError(f"Unsupported file type for {filename}")
    return df

def attempt_common_encodings(filename):
    common_encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1252', 'iso-8859-1']
    for encoding in common_encodings:
        try:
            return load_data(filename, encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not determine the correct encoding for the file.")

def convert_date_columns(df):
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except ValueError:
                logging.warning(f"Warning: Could not convert column '{col}' to datetime.")
    return df

def apply_query(data, condition):
    parsed = sqlparse.parse(condition)[0]
    
    try:
        if parsed.get_type() == 'SELECT':
            from_clause = "**********"
            if from_clause:
                table_name = from_clause.get_name()
                if table_name.lower() != 'data':
                    raise ValueError(f"Table '{table_name}' not recognized. Only 'data' is supported.")
            
            select_clause = "**********"
            columns = [str(col) for col in select_clause.get_identifiers()] if select_clause else ['*']
            
            where_clause = "**********"
            if where_clause:
                condition_str = str(where_clause).replace('WHERE ', '')
                result = data.query(condition_str)
            else:
                result = data
            
            if '*' in columns:
                columns = data.columns.tolist()
            result = result[columns]

            # Handle GROUP BY
            groupby_clause = "**********"
            if groupby_clause:
                groupby_columns = [str(col) for col in groupby_clause.get_identifiers()]
                result = result.groupby(groupby_columns).agg(list)  # Placeholder for aggregation

            # Handle ORDER BY
            orderby_clause = "**********"
            if orderby_clause:
                order_columns = []
                for order in orderby_clause.get_identifiers():
                    col = str(order)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"o "**********"r "**********"d "**********"e "**********"r "**********". "**********"t "**********"t "**********"y "**********"p "**********"e "**********"  "**********"i "**********"s "**********"  "**********"s "**********"q "**********"l "**********"p "**********"a "**********"r "**********"s "**********"e "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********". "**********"K "**********"e "**********"y "**********"w "**********"o "**********"r "**********"d "**********". "**********"O "**********"r "**********"d "**********"e "**********"r "**********": "**********"
                        if col.upper() == 'DESC':
                            order_columns.append((str(orderby_clause.get_identifiers()[0]), False))
                        else:
                            order_columns.append((str(orderby_clause.get_identifiers()[0]), True))
                result = result.sort_values(by=[col for col, _ in order_columns], ascending=[asc for _, asc in order_columns])

            # Handle LIMIT
            limit_clause = "**********"
            if limit_clause:
                limit_value = "**********"
                if limit_value:
                    result = result.head(int(limit_value.value))

        else:
            # Handle simple conditions or other query types if necessary
            conditions = condition.split(' AND ')
            result = data
            
            for cond in conditions:
                if ' == ' in cond:
                    column, value = cond.split(' == ', 1)
                    op = '=='
                elif ' != ' in cond:
                    column, value = cond.split(' != ', 1)
                    op = '!='
                elif ' > ' in cond:
                    column, value = cond.split(' > ', 1)
                    op = '>'
                elif ' < ' in cond:
                    column, value = cond.split(' < ', 1)
                    op = '<'
                elif ' >= ' in cond:
                    column, value = cond.split(' >= ', 1)
                    op = '>='
                elif ' <= ' in cond:
                    column, value = cond.split(' <= ', 1)
                    op = '<='
                elif ' LIKE ' in cond:
                    column, value = cond.split(' LIKE ', 1)
                    op = 'LIKE'
                else:
                    raise ValueError(f"Unsupported operator in condition: {cond}")
                
                column = column.strip()
                value = value.strip().strip("'").strip('"')
                
                col_type = data[column].dtype
                if col_type.name.startswith('datetime64'):
                    value = pd.to_datetime(value, errors='coerce')
                    if pd.isna(value):
                        raise ValueError(f"Invalid date format for value '{value}' in condition for column '{column}'")
                elif col_type.name.startswith('int'):
                    value = int(value)
                elif col_type.name.startswith('float'):
                    value = float(value)
                elif col_type == 'bool':
                    value = value.lower() == 'true'
                
                if op == '>':
                    result = result[result[column] > value]
                elif op == '<':
                    result = result[result[column] < value]
                elif op == '>=':
                    result = result[result[column] >= value]
                elif op == '<=':
                    result = result[result[column] <= value]
                elif op == '==':
                    result = result[result[column] == value]
                elif op == '!=':
                    result = result[result[column] != value]
                elif op == 'LIKE':
                    result = result[result[column].astype(str).str.contains(value.replace('%', '.*'), regex=True, case=False)]
                else:
                    raise ValueError(f"Unsupported operator: {op}")

    except Exception as e:
        logging.error(f"Query parsing or execution error: {e}")
        return None

    logging.info(result.head())
    return result