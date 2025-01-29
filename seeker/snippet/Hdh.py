#date: 2025-01-29T16:59:31Z
#url: https://api.github.com/gists/3848d6bc24919fedcdcc07f6979ebd96
#owner: https://api.github.com/users/VidhyaVarshanyJS

import pandas as pd

import numpy as np

from functools import lru_cache

import multiprocessing as mp

from typing import List, Dict, Set, Tuple

from pathlib import Path

import concurrent.futures

@lru_cache(maxsize=1024)

def load_mrconso_data(mrconso_path: str) -> pd.DataFrame:

    """

    Load and cache MRCONSO.RRF data with specific columns.

    Only loads English language entries.

    """

    columns = ['CUI', 'LAT', 'ISPREF', 'STR', 'SAB']

    df = pd.read_csv(mrconso_path, sep='|', header=None, usecols=[0, 1, 6, 14, 11],

                     names=columns, dtype=str)

    return df[df['LAT'] == 'ENG']

@lru_cache(maxsize=1024)

def load_mrsty_data(mrsty_path: str) -> pd.DataFrame:

    """

    Load and cache MRSTY.RRF data.

    """

    columns = ['CUI', 'TUI', 'STN', 'STY']

    df = pd.read_csv(mrsty_path, sep='|', header=None, usecols=[0, 1, 2, 3],

                     names=columns, dtype=str)

    return df

def get_cuis_for_term(term: str, mrconso_df: pd.DataFrame) -> List[str]:

    """

    Get all CUIs for a given term (case-insensitive).

    """

    return mrconso_df[mrconso_df['STR'].str.lower() == term.lower()]['CUI'].unique().tolist()

def get_semantic_types(cui: str, mrsty_df: pd.DataFrame, allowed_semantic_types: Set[str]) -> List[str]:

    """

    Get semantic types for a CUI that are in the allowed list.

    """

    semantic_types = mrsty_df[mrsty_df['CUI'] == cui]['STY'].unique().tolist()

    return [st for st in semantic_types if st in allowed_semantic_types]

def get_preferred_name(cui: str, mrconso_df: pd.DataFrame) -> str:

    """

    Get preferred name for a CUI.

    """

    pref_names = mrconso_df[(mrconso_df['CUI'] == cui) & 

                           (mrconso_df['ISPREF'] == 'Y')]['STR'].tolist()

    return pref_names[0] if pref_names else ''

def get_atom_list(cui: str, mrconso_df: pd.DataFrame) -> List[str]:

    """

    Get all atom names (STR) for a CUI.

    """

    return mrconso_df[mrconso_df['CUI'] == cui]['STR'].unique().tolist()

def process_row(row: pd.Series, mrconso_df: pd.DataFrame, mrsty_df: pd.DataFrame, 

               allowed_semantic_types: Set[str]) -> List[Dict]:

    """

    Process a single row and return a list of dictionaries with expanded CUI information.

    """

    nkt_term = str(row['NKT']).strip()

    if pd.isna(nkt_term) or not nkt_term:

        return []

    

    results = []

    cuis = get_cuis_for_term(nkt_term, mrconso_df)

    

    for cui in cuis:

        semantic_types = get_semantic_types(cui, mrsty_df, allowed_semantic_types)

        if not semantic_types:  # Skip if no matching semantic types

            continue

            

        pref_name = get_preferred_name(cui, mrconso_df)

        atom_list = get_atom_list(cui, mrconso_df)

        

        # Create new row for each CUI

        new_row = row.to_dict()

        new_row.update({

            'CUIS': cui,

            'Semantic type': '|'.join(semantic_types),

            'Pref Name': pref_name if pref_name != nkt_term else nkt_term,

            'Atom List': '|'.join(atom_list)

        })

        results.append(new_row)

    

    return results

def process_chunk(args: Tuple) -> List[Dict]:

    """

    Process a chunk of rows for parallel processing.

    """

    chunk, mrconso_path, mrsty_path, allowed_semantic_types = args

    mrconso_df = load_mrconso_data(mrconso_path)

    mrsty_df = load_mrsty_data(mrsty_path)

    

    results = []

    for _, row in chunk.iterrows():

        results.extend(process_row(row, mrconso_df, mrsty_df, allowed_semantic_types))

    return results

def main():

    # File paths - update these

    input_xlsx = "input.xlsx"

    mrconso_path = "MRCONSO.RRF"

    mrsty_path = "MRSTY.RRF"

    output_xlsx = "output.xlsx"

    

    # Define allowed semantic types

    allowed_semantic_types = {

        "Disease or Syndrome",

        "Sign or Symptom",

        # Add other allowed semantic types here

    }

    

    # Read input Excel file

    df = pd.read_excel(input_xlsx)

    

    # Split dataframe into chunks for parallel processing

    num_chunks = mp.cpu_count()

    chunks = np.array_split(df, num_chunks)

    

    # Prepare arguments for parallel processing

    chunk_args = [(chunk, mrconso_path, mrsty_path, allowed_semantic_types) 

                 for chunk in chunks]

    

    # Process chunks in parallel

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        chunk_results = executor.map(process_chunk, chunk_args)

        for chunk_result in chunk_results:

            results.extend(chunk_result)

    

    # Convert results to DataFrame and save

    result_df = pd.DataFrame(results)

    result_df.to_excel(output_xlsx, index=False)

if __name__ == "__main__":

    main()