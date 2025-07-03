#date: 2025-07-03T17:11:28Z
#url: https://api.github.com/gists/2c496a2d10f33f8bcec19648cbd41026
#owner: https://api.github.com/users/cmdcolin

import os
import re
import sys

# Check if command line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <input_file> [output_directory]")
    sys.exit(1)

input_file = sys.argv[1]

# Use current directory by default, or the specified directory if provided
output_directory = os.getcwd()
if len(sys.argv) > 2:
    output_directory = sys.argv[2]

with open(input_file, "r") as f:
    content = f.read()

blocks = content.split("\n//\n")

for block in blocks:
    if not block.strip():
        continue

    accession_match = re.search(r"^#=GF AC\s+(\S+)", block, re.MULTILINE)
    if accession_match:
        filename = accession_match.group(1) + ".txt"
        output_path = os.path.join(output_directory, filename)
        with open(output_path, "w") as out_f:
            out_f.write(block.strip() + "\n//\n")
    else:
        print(
            f"Warning: Could not find #=GF AC in a block. Skipping block: {block[:50]}..."
        )
