#date: 2025-04-25T16:55:39Z
#url: https://api.github.com/gists/8c5ebf09d4e5c62059a9e8aed3b4b6a5
#owner: https://api.github.com/users/SWORDIntel

#!/usr/bin/env python3
"""
sort_credentials.py

TUI-style splitter for EMAIL:PASS dumps.  

• Lists all files in cwd and prompts you to pick one.  
• Creates Sortedmails/ and, for each domain, .txt, .csv, .json.  
• Supports resume via a hidden state file.  
• Shows a tqdm progress bar so you know it’s alive.  
• Regenerates JSON arrays at the end to keep them valid.
"""

import os
import sys
import logging
import json
from glob import glob

# you’ll need tqdm: `pip3 install tqdm`
from tqdm import tqdm

def select_file():
    scripts = os.path.basename(__file__)
    candidates = [
        f for f in os.listdir('.') 
        if os.path.isfile(f) 
        and f != scripts 
        and not f.startswith('Sortedmails')
    ]
    if not candidates:
        print("❌ No files found to process in this directory.")
        sys.exit(1)
    print("\nSelect a file to process:\n")
    for i, fn in enumerate(candidates, 1):
        print(f"  {i}. {fn}")
    choice = input(f"\nEnter number [1–{len(candidates)}]: ").strip()
    try:
        idx = int(choice) - 1
        return candidates[idx]
    except:
        print("❌ Invalid selection.")
        sys.exit(1)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    infile = select_file()
    outdir = 'Sortedmails'
    os.makedirs(outdir, exist_ok=True)

    # state file to remember where we left off
    state_file = os.path.join(outdir, f".{infile}.state")

    # count total lines
    logging.info(f"Counting total lines in {infile}…")
    with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f)

    # decide where to start
    start = 0
    if os.path.exists(state_file):
        ans = input(f"Resume from last run? [y/N]: ").strip().lower()
        if ans == 'y':
            try:
                start = int(open(state_file).read().strip())
            except:
                start = 0

    logging.info(f"Processing {total_lines} lines, starting at line {start}.")

    # open handles per-domain (only txt & csv; json is built at the end)
    txt_handles = {}
    csv_handles = {}

    try:
        with open(infile, 'r', encoding='utf-8', errors='ignore') as fin:
            for lineno, raw in enumerate(
                tqdm(fin, total=total_lines, initial=start, desc="Sorting")
            ):
                if lineno < start:
                    continue
                line = raw.strip()
                if not line or ':' not in line:
                    continue

                email, password = line.split(': "**********"
                if '@' not in email:
                    continue
                domain = email.split('@',1)[1]

                # first time seeing this domain?
                if domain not in txt_handles:
                    txt_p = os.path.join(outdir, f"{domain}.txt")
                    csv_p = os.path.join(outdir, f"{domain}.csv")

                    txt_handles[domain] = open(txt_p, 'a', encoding='utf-8')
                    new_csv = not os.path.exists(csv_p) or os.path.getsize(csv_p) == 0
                    csv_handles[domain] = open(csv_p, 'a', encoding='utf-8')
                    if new_csv:
                        # write header
                        csv_handles[domain].write("email,password\n")

                # append to files
                txt_handles[domain].write(f"{email}: "**********"
                csv_handles[domain].write(f"{email},{password}\n")

                # checkpoint
                with open(state_file, 'w') as sf:
                    sf.write(str(lineno + 1))

    finally:
        # clean up
        for h in txt_handles.values():
            h.close()
        for h in csv_handles.values():
            h.close()

    logging.info("Text/CSV split done → generating JSON outputs…")

    # regenerate all JSON per-domain from CSV
    csv_list = glob(os.path.join(outdir, "*.csv"))
    for csv_f in tqdm(csv_list, desc="JSON"):
        domain = os.path.splitext(os.path.basename(csv_f))[0]
        json_p = os.path.join(outdir, f"{domain}.json")
        data = []
        with open(csv_f, 'r', encoding='utf-8') as cf:
            next(cf)  # skip header
            for row in cf:
                e,p = row.strip().split(',',1)
                data.append({"email": "**********": p})
        with open(json_p, 'w', encoding='utf-8') as jf:
            json.dump(data, jf, indent=2)

    logging.info("All done! ✅")
    print(f"\nFiles written under ./{outdir}/\n")

if __name__ == '__main__':
    main()
