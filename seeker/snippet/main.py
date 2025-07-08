#date: 2025-07-08T16:51:17Z
#url: https://api.github.com/gists/a6db0252168d0747b40523c65e27b377
#owner: https://api.github.com/users/N11cc00

import os
import re


RIOT_DIR = "/home/nico/dev/riot/drivers"

ignore = "/vendor/"

skipped_files = []

# Simple mapping from license keywords to SPDX identifiers
LICENSE_KEYWORDS = {
    "gnu lesser general public license v2.1": "LGPL-2.1-only",
    "gnu general public license v2": "GPL-2.0-only",
    "gnu general public license v3": "GPL-3.0-only",
    "mit license": "MIT",
    "apache license 2.0": "Apache-2.0",
    "bsd license": "BSD-2-Clause",  # or BSD-3-Clause depending on specifics
    # add more as needed
}



def extract_clean_copyright(filepath):
    # Match the comment block at the start of the file (including multiline)
    pattern = re.compile(r"(?s)/\*(.*?)\*/")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = pattern.match(content)
    if not match:
        return None

    # Extract the inner content of the comment block
    comment_body = match.group(1)

    # Remove leading '*' and whitespace from each line
    lines = comment_body.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove leading spaces, then optional '*', then optional space after '*'
        cleaned_line = re.sub(r"^\s*\*\s?", "", line)
        cleaned_lines.append(cleaned_line)

    # Join lines back into a single string
    clean_text = "\n".join(cleaned_lines).strip()
    return clean_text


def detect_license(text):
    """Simple license detection by keyword matching (case-insensitive)."""
    text_lower = text.lower()

    # replace newlines with spaces for the check here

    text_lower_continous = text_lower.replace("\n", " ")
    for keyword, spdx_id in LICENSE_KEYWORDS.items():
        if keyword in text_lower_continous:
            return spdx_id
    return "NONE"  # fallback if no license detected

def build_spdx_header(years, holders, license_id):
    assert years != []
    assert holders != []
    assert len(years) == len(holders)

    spdx_texts = []

    for year, holder in zip(years, holders):
        assert not "2" in holder
        assert "2" in str(year)
        assert not ("All rights" in holder)

        spdx_texts.append(f" * SPDX-FileCopyrightText: {year} {holder}")


    spdx_header = "/*\n"

    for spdx_text in spdx_texts:
        spdx_header += (spdx_text + "\n")

    spdx_header += f" * SPDX-License-Identifier: {license_id}\n"


    spdx_header += " */\n"

    return spdx_header

def parse_copyrights(text):
    """Parse copyright statements in the text, return sets of years and holders."""
    # Simple regex to capture lines like "Copyright (C) 2018 Inria"
    # Captures year(s) and holder name


    """
    copyright_re = re.compile(
    r"copyright(?: \(c\))?\s*(?P<years>(?:\d{4}(?:-\d{4})?(?:, )?)+)?\s*(?P<holder>.+)", 
    re.I
    )
    """
    copyright_re = re.compile(
        r"copyright(?: \(c\))?\s*(?P<years>(?:\d{4}(?:-\d{4})?)+)?\s*(?P<holder>.+)", 
        re.I
    )
    years = []
    holders = []

    for line in text.splitlines():
        m = copyright_re.search(line)
        if m:
            year_str = m.group("years")
            if year_str:
                # Split multiple years if present, e.g. "2017, 2018-2020"
                parts = re.split(r",\s*", year_str)
                for part in parts:
                    years.append(part.strip())
            holder = m.group("holder").strip()
            holders.append(holder)

    return years, holders

def convert_file(filepath):
    print(f"Processing {filepath}")
    clean_text = extract_clean_copyright(filepath)
    if not clean_text:
        print(f"No copyright header found in {filepath}")
        return
    years, holders = parse_copyrights(clean_text)
    license_id = detect_license(clean_text)

    try:
        spdx_header = build_spdx_header(years, holders, license_id)
    except AssertionError as e:
        print(f"Skipping: {str(e)}")
        skipped_files.append(filepath)
        return
    
    replace_header(filepath, spdx_header)
    
    print(f"{spdx_header}")

def replace_header(file_path: str, new_header: str) -> None:
    """
    Replace the initial C-style /* ... */ comment block at the top of the file with a new header.

    Parameters:
        file_path (str): Path to the source file to be modified.
        new_header (str): New header to insert, should include /* and */.
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match a /* ... */ comment block at the top of the file
    pattern = r'^\s*/\*.*?\*/\s*'  # includes leading/trailing whitespace
    new_header_clean = new_header.strip() + '\n\n'

    if re.match(pattern, content, flags=re.DOTALL):
        # Replace the existing top block comment
        new_content = re.sub(pattern, new_header_clean, content, count=1, flags=re.DOTALL)
    else:
        # No header found, just prepend the new header
        new_content = new_header_clean + content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    # Walk through the RIOT source tree
    for root, _, files in os.walk(RIOT_DIR):
        for file in files:
            if ignore in os.path.join(root, file):
                print(f"Ignoring {os.path.join(root, file)}")
                continue

            if file.endswith(".h")  or file.endswith(".c"):
                print("-----------")
                convert_file(os.path.join(root, file))
                print("-----------")
            # else:
            #     print(f"Invalid file type: {os.path.join(root, file)}")
    
    print(skipped_files)
    print(len(skipped_files))

