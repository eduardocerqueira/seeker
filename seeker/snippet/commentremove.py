#date: 2024-05-30T16:55:10Z
#url: https://api.github.com/gists/5ef03d9810f7f00c4c4e54bebe32379c
#owner: https://api.github.com/users/tjdwill

# The Python version
# A script to condense Rust source files by removing line comments.

"""
- Load file
- Read each line; ignore whitespace up to first non-ws character
- Check for comment signal
- Ignore that line
- Write non-comment lines to file.
- Keep count of lines written
"""

import argparse
from pathlib import Path

# Helpful setup

parser = argparse.ArgumentParser()
parser.add_argument("src", help="The Rust source file.")

WHITESPACE = (' ', '\t')
COMMENT_TKN = '//'
cat = "".join

if __name__ == "__main__":
    
    src = Path(parser.parse_args().src)
    if not src.suffix == ".rs":
        raise ValueError("Input must be a Rust source file.")
    
    outfile = src.parent / cat([src.stem, "_strp", ".rs"])

    with open(src) as full, open(outfile, 'w') as stripped:
        lineno = 0
        while True:
            line = full.readline()
            if line.lstrip(cat(WHITESPACE))[:2] == COMMENT_TKN:
                continue
            elif line == "": 
                print(f"Lines written: {lineno}")
                break
            else:
                stripped.write(line)
                lineno += 1