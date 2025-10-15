#date: 2025-10-15T17:13:06Z
#url: https://api.github.com/gists/3a79beac141ebaf26b629d3182929851
#owner: https://api.github.com/users/rma92

#!/usr/bin/python
#Install Pandoc, and pip install mwxml mwparserfromhell tqdm pypandoc

import os
import re
import mwxml
import mwparserfromhell
from tqdm import tqdm

# Input / output
INPUT_DUMP = "Bulbapedia-20251014170426.xml"
OUTPUT_DIR = "bulba"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sanitize filenames so Windows doesn't complain
def safe_filename(title: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "_", title)
    return clean[:150].strip() or "untitled"

# Open the dump
print(f"Reading {INPUT_DUMP} ...")
dump = mwxml.Dump.from_file(open(INPUT_DUMP, "r", encoding="utf-8"))

# Iterate through pages and write latest revision of each
for page in tqdm(dump, desc="Extracting pages"):
    try:
        for rev in page:
            text = rev.text or ""
            #plain = mwparserfromhell.parse(text).strip_code().strip()
            plain = mwparserfromhell.parse(text).strip_code().strip()
            if not plain:
                break
            filename = safe_filename(page.title) + ".txt"
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write(f"# {page.title}\n\n{plain}\n")
            break  # only latest revision
    except Exception as e:
        print(f" Error on page '{page.title}': {e}")

print(f"Done! Extracted pages to .\\{OUTPUT_DIR}\\")

