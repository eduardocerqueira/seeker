#date: 2024-01-10T17:06:56Z
#url: https://api.github.com/gists/f80bb7aaf7638e3013f8a26c6ae21ccd
#owner: https://api.github.com/users/orellabac

# How to use:
#   1. In VS Code, perform a search.
#   2. Click "Open in editor" to open the search results in a `.code-search` file
#   3. The Save the file
#   4. In terminal, run `python export-vscode-search-to-csv.py  search-results.code-search path/to/exported.csv`

import csv
import sys
import os
import re

# Constants
FILENAME_REGEX = re.compile(r'^([^\s+].*?):$')
LINE_REGEX = re.compile(r'^\s+(\d+):\s*(.*?)\s*$')

def escape_string(s):
    return f'"{s}"'

def log_error(msg, log_type=print, code=1):
    log_type(msg)
    sys.exit(code)

# Parsing

cmd, script, *args = sys.argv

if not args:
    log_error(f"Usage: {cmd} {script} /path/to/input/file.code-search /path/to/output/file")

input_file, output_file = args
extension = os.path.splitext(input_file)[1]

if extension != ".code-search":
    log_error(f"ERROR: {extension} not supported. Supported extensions:\n\t.code-search")

if not output_file:
    log_error(f"ERROR: you must provide an output file.\n\t{cmd} {script} {input_file} /path/to/output/file", code=1)

if os.path.exists(output_file):
    log_error(f"ERROR: {output_file} already exists! Please remove it and try again.", code=1)

# Set up streams
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # write header row
    writer.writerow(["Path", "Line number", "Result"])

    # Set up read stream
    current_file = None
    count = 0

    with open(input_file, 'r') as f:
        for line in f:
            if FILENAME_REGEX.match(line):
                current_file = FILENAME_REGEX.match(line).group(1)
            elif LINE_REGEX.match(line):
                match = LINE_REGEX.match(line)
                line_number, result = match.group(1), match.group(2)
                if line_number and result:
                    writer.writerow([escape_string(current_file), escape_string(line_number), escape_string(result)])
                    count += 1

print(f"Done! Wrote {count} rows to {output_file}")