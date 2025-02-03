#date: 2025-02-03T16:55:15Z
#url: https://api.github.com/gists/c8e126505e7545d25e17ab0c7c8ddb8b
#owner: https://api.github.com/users/VidhyaVarshanyJS

import pandas as pd
import mmap
import concurrent.futures

# File Paths
MRCONSO_FILE = "MRCONSO.RRF"
MRSTY_FILE = "MRSTY.RRF"

# Load valid semantic types (adjust this based on your actual list)
VALID_SEMANTIC_TYPES = {"T047", "T191", "T123"}

# Memory map and read MRSTY to create a dictionary (CUI → SEM_Type)
def load_semantic_types():
    sty_dict = {}
    with open(MRSTY_FILE, "r", encoding="utf-8") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for line in iter(mm.readline, b""):
            parts = line.decode("utf-8").strip().split("|")
            sty_dict[parts[0]] = parts[1]  # CUI → Semantic Type
    return sty_dict

semantic_type_map = load_semantic_types()

# Stream process MRCONSO using mmap
def process_mrconso():
    with open(MRCONSO_FILE, "r", encoding="utf-8") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for line in iter(mm.readline, b""):
            parts = line.decode("utf-8").strip().split("|")
            yield parts[0], parts[14]  # CUI, Atom_Name

# Convert to a fast lookup set
atom_names = set(atom for _, atom in process_mrconso())

# Function to process a row
def process_row(data):
    cui, atom_name = data
    if "+" in atom_name:
        parts = atom_name.split("+")
        separate_atom_exists = all(part in atom_names for part in parts)
        is_semantic_valid = semantic_type_map.get(cui, "INVALID") in VALID_SEMANTIC_TYPES
        return cui, atom_name, separate_atom_exists, is_semantic_valid

# Process MRCONSO in parallel
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in executor.map(process_row, process_mrconso()):
        if result:
            results.append(result)

# Save to CSV
pd.DataFrame(results, columns=["CUI", "Atom_Name", "Separate_Atom_Exist", "Is_Semantic_Valid"]).to_csv("filtered_mrconso.csv", index=False)
