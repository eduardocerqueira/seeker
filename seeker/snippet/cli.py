#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import click
from ingest.ingest import ingest_data
from query.query import detect_encoding_and_query
from visualize.visualize import visualize_user_activity_by_date, visualize_chat_frequency, visualize_online_status, visualize_activity_heatmap
from src.nlp.nlp_query.nlp_query import handle_nlp_query
from src.nlp.nlp_ingest.nlp_ingest import handle_nlp_ingest
from src.nlp.nlp_visualize.nlp_visualize import handle_nlp_visualize
from tqdm import tqdm
import time
import os
import pandas as pd

@click.group()
def cli():
    """ELD - Extremely Large Data handling tool"""
    pass

def validate_file_path(filename):
    """Validate if the file exists and return the absolute path."""
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return filename

@cli.command()
@click.argument('query', nargs=-1)
def ingest(query):
    """Ingest data with a natural language query.
    
    Examples:
        eld ingest "Load data from source file.csv into my_table"
        eld ingest "Import data from data.json"
    """
    query_string = ' '.join(query)
    try:
        with tqdm(total=100, desc="Ingesting data") as pbar:
            result = handle_nlp_ingest(query_string)
            for _ in range(10):
                time.sleep(0.1)
                pbar.update(10)
        if isinstance(result, pd.DataFrame):
            print(f"Data ingestion interpreted as: {query_string}")
            print(result.head())  # Assuming result is a DataFrame
        elif isinstance(result, str):
            print(result)  # This would print any string message from the ingestion process
        else:
            print("Data ingestion failed. Please check your query.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

@cli.command()
@click.argument('filename')
@click.argument('query', nargs=-1)
@click.option('--page', type=int, default=1, help='Page number of results to display')
@click.option('--per-page', type=int, default=10, help='Number of items per page')
def query(filename, query, page, per_page):
    """Query data with a natural language or SQL-like query.
    
    Examples:
        eld query data.csv "SELECT * FROM data WHERE date_joined > '2020-01-01'"
        eld query data.json "Show users who joined after 2020-01-01 and are online"
    """
    try:
        filename = validate_file_path(filename)
    except FileNotFoundError as e:
        print(e)
        return

    query_string = ' '.join(query)
    try:
        with tqdm(total=100, desc="Querying data") as pbar:
            if query_string.startswith(('SELECT', 'select')):  # SQL-like query
                result = detect_encoding_and_query(filename, query_string)
            else:  # Natural language query
                sql_like = handle_nlp_query(query_string)
                print(f"Generated SQL-like query: {sql_like}")
                result = detect_encoding_and_query(filename, sql_like)
            for _ in range(10):
                time.sleep(0.1)
                pbar.update(10)
        if result is not None:
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            print(result.iloc[start_idx:end_idx])
            total_pages = (len(result) + per_page - 1) // per_page
            print(f"Showing page {page} of {total_pages}.")
        else:
            print("Query returned no results or encountered an error.")
    except Exception as e:
        print(f"Error during query: {e}")

@cli.group()
def visualize():
    """Visualize data."""
    pass

@visualize.command()
@click.argument('query', nargs=-1)
@click.argument('filename')
def natural(query, filename):
    """Visualize data with a natural language query.
    
    Examples:
        eld visualize natural "Show activity heatmap by day" data.csv
        eld visualize natural "Display chat frequency by user" data.json"
    """
    try:
        filename = validate_file_path(filename)
    except FileNotFoundError as e:
        print(e)
        return
    query_string = ' '.join(query)
try:
    with tqdm(total=100, desc="Visualizing data") as pbar:
        data = detect_encoding_and_query(filename, '')  # Load data without applying a query
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)
        if data is not None:
            result = handle_nlp_visualize(query_string, data)
            if result:
                print(f"Visualization based on: {query_string}")
                print(result)
            else:
                print("Visualization not understood or not implemented yet.")
        else:
            print("Error: Data could not be loaded.")
except Exception as e:
    print(f"Error during visualization: {e}")

@visualize.command()
@click.argument('filename')
@click.option('--type', type=click.Choice(['activity', 'chat', 'status', 'heatmap']), default='activity', help='Type of visualization')
def type(filename, type):
    """Visualize data from the file with a specific type.
    
    Examples:
        eld visualize type data.csv --type activity
        eld visualize type data.json --type chat
    """
    try:
        filename = validate_file_path(filename)
    except FileNotFoundError as e:
        print(e)
        return

    try:
        with tqdm(total=100, desc="Visualizing data") as pbar:
            data = detect_encoding_and_query(filename, '')  # Load data without applying a query
            for _ in range(10):
                time.sleep(0.1)
                pbar.update(10)
            if data is None:
                print("Error: Data could not be loaded.")
                return
            if type == 'activity':
                visualize_user_activity_by_date(data)
            elif type == 'chat':
                visualize_chat_frequency(data)
            elif type == 'status':
                visualize_online_status(data)
            elif type == 'heatmap':
                visualize_activity_heatmap(data)
            print(f"Visualization saved as {type}_visualization.png")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    cli()