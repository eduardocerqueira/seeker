#date: 2021-11-25T17:15:33Z
#url: https://api.github.com/gists/47d279f06744c55dd3327ce1dcb7c4d5
#owner: https://api.github.com/users/weerasuriya

#!/usr/bin/env python

"""
MIT License
Copyright 2020 Joe Bentley

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Process the bibliography, performing the following three actions:
1. Shorten author lists to 10 authors
2. Abbreviate journal names
3. Remove unneeded IDs and replace with doi links
4. Remove unneeded tags (file, abstract)

Required bibtexparser, install using "pip install bibtexparser"

Use with "./process-bibliography.py [filename]"
"""

import bibtexparser
import sys
import os

filepath = sys.argv[1]

# truncate author lists greater than 10 to 3 authors
def truncate_to_three_authors(authors_string):
    authors = authors_string.split('and')
    if len(authors) > 10:  # et.al.
        authors_truncated = 'and'.join(authors[:3]) + "and others"
        return authors_truncated
    else:
        return authors_string

# Set of journal abbreviations (case-sensitive replacement)
# ADD YOURS HERE
journal_abbreviations = {
    "Physical Review A": "Phys. Rev. A",
    "Physical Review A - Atomic, Molecular, and Optical Physics": "Phys. Rev. A",
    "Physical Review D": "Phys. Rev. D",
    "Physical Review D - Particles, Fields, Gravitation and Cosmology": "Phys. Rev. D",
    "Physical Review Letters": "Phys. Rev. Lett. ",
    "Classical and Quantum Gravity": "Class. Quantum Grav. ",
    "Journal of Physics B: Atomic, Molecular and Optical Physics":
    "J. Phys. B: At. Mol. Opt. Phys. ",
    "Journal of Lightwave Technology": "J. Lightwave Technol. ",
    "Reviews of Modern Physics": "Rev. Mod. Phys. ",
    "Optics Express": "Opt. Express",
    "Optics Communications": "Opt. Commun. ",
    "Nature Physics": "Nat. Phys. ",
    "Nature Photonics": "Nat. Photonics",
    "Nature Astronomy": "Nat. Astron. ",
    "Nuclear Instruments and Methods in Physics Research, Section A: Accelerators, Spectrometers, Detectors and Associated Equipment":
    "Nucl. Instrum. Meth. A",
    "Reports on Progress in Physics": "Rep. Prog. Phys. ",
    "IEEE Transactions on Automatic Control": "IEEE T. Automat. Contr. ",
    "IEEE Transactions on Circuits and Systems": "IEEE T. Circuits Syst. ",
    "IEEE 55th Conference on Decision and Control": "IEEE Decis. Contr. P. ",
    "Information and Control": "Inform. Control",
    "SIAM Journal on Control and Optimization": "SIAM J. Control. Optim. ",
    "Communications in Mathematical Physics": "Comm. Math. Phys. ",
    "Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences": "Philos. T. R. Soc. A",
    "Journal of the Optical Society of America": "J. Opt. Soc. Am. ",
    "Journal of the Electrochemical Society": "J. Electrochem. Soc. ",
    "Advanced in Physics: X": "Adv. Phys. X",
    "Science China: Physics, Mechanics and Astronomy": "Sci. China Phys. Mech. "
}

# Abbreviate journal names case sensitively
def abbreviate_journal_name(journal_name):
    journal_name = journal_name.strip()
    if journal_name in journal_abbreviations:
        return journal_abbreviations[journal_name]
    else:
        return journal_name

# Remove archiveprefix, arxivid, eprint keys
# Only use first of the URLs listed
# If there is no URL, construct a DOI URL link
def remove_ids_and_fix_links(entry):
    if 'archiveprefix' in entry:
        del entry['archiveprefix']
    if 'arxivid' in entry:
        del entry['arxivid']
    if 'eprint' in entry:
        del entry['eprint']

    # only use first of the URLs listed
    if 'url' in entry:
        split_urls = entry['url'].split(' ')
        if len(split_urls) > 1:
            entry['url'] = split_urls[0]
    
    if 'doi' in entry:
        if 'url' not in entry:
            doi = entry['doi']
            entry['url'] = f"https://doi.org/{doi}"
        del entry['doi']


with open(filepath) as f:
    # load the bibtex file
    bib_database = bibtexparser.load(f)

    # each entry is a bibtex entry
    for entry in bib_database.entries:
        entry['author'] = truncate_to_three_authors(entry['author'])

        if 'journal' in entry:
            entry['journal'] = abbreviate_journal_name(entry['journal'])

        remove_ids_and_fix_links(entry)

        if 'abstract' in entry:
            del entry['abstract']
        if 'file' in entry:
            del entry['file']

    # save the bibtex file
    filepath, ext = os.path.splitext(filepath)
    filepath = f"{filepath}-processed{ext}"

    with open(filepath, 'w') as output_file:
        bibtexparser.dump(bib_database, output_file)
        print(f"Wrote to {filepath}")
