#date: 2026-02-13T17:27:51Z
#url: https://api.github.com/gists/fbc43cd71682ce87aa96e8cda6d6bf99
#owner: https://api.github.com/users/pmcneely

import os
import time
from pathlib import Path

RELEASE_DATE = "2026-02-05"
RELEASE_DIR = f"releases/{RELEASE_DATE}/nt/"
BASE_PATH = Path(__file__).parent

nt_path = os.path.join(BASE_PATH, RELEASE_DIR)
print(nt_path)

concatenated_lines = []

start = time.time()
for root, dirs, files in os.walk(nt_path):
    if root != nt_path:
        continue
    s_files = sorted(files)
    for f in s_files:
        with open(os.path.join(root, f), "r") as fp:
            lines = fp.readlines()
        concatenated_lines.extend(lines)
print(f"Found {len(concatenated_lines)} tuples")

with open("biomarkerKG.nt", "w") as output_p:
    for line in concatenated_lines:
        # nt format requires trailing '.\n' construct
        # corrected_line = " ".join(line.split(" ")[:-1])
        output_p.write(line)
print(f"Processing needed {time.time() - start:.2f} seconds")