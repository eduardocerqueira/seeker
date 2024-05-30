#date: 2024-05-30T17:01:57Z
#url: https://api.github.com/gists/2b1b71d0f6f7f96614d8a90ebdbe68a6
#owner: https://api.github.com/users/tjdwill

# Increase the Bookmark Level of a given PDF
# This is a quick script to make the addition of top-level marks easier.
import argparse
from pathlib import Path


supported_extensions = [".info"]

parser = argparse.ArgumentParser()
parser.add_argument("src_file", help="The .info file for the pdf.")

cat = "".join


if __name__ == "__main__":
    # Get file name
    args = parser.parse_args()
    src = Path(args.src_file)

    if not src.is_file():
        raise FileExistsError
    elif src.suffix not in supported_extensions:
        raise ValueError(f"Unsupported file extension: {src.suffix}")
    
    # Setup output
    outfile = src.parent / cat(["mod_", src.name])

    with open(src, mode='r', encoding="utf-8") as fin, open(outfile, mode='w', encoding="utf-8") as fout:
        # Iterate through each line, modifying those with BookmarkLevel 
        while True:
            line = fin.readline()
            if line == "":
                break
            if line.startswith("BookmarkLevel"):
                parts = line.split()
                num = int(parts[1])
                parts[1] = str(num+1)
                parts.append('\n')
                outline = " ".join(parts) 
                fout.write(outline)
            else:
                fout.write(line)
