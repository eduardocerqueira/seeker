#date: 2025-01-15T16:51:22Z
#url: https://api.github.com/gists/abc4f2b30c695d4e6b9cec51d1622438
#owner: https://api.github.com/users/dylan-k

import csv

def generate_bibtex_key(authors, year):
    """Generate a BibTeX key from the first author's last name and the year"""
    last_names = authors.split(",")[0].split()
    last_name = last_names[-1] if len(last_names) > 0 else ""
    return f"{last_name.lower()}{year}"

def convert_csv_to_bibtex(csv_file, bibtex_file):
    """Convert a CSV file to BibTeX format"""
    with open(csv_file, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)

        with open(bibtex_file, "w", encoding="utf-8") as bibtexfile:
            for row in reader:
                authors = row["Authors"]
                year = row["Year"]
                title = row["Title"]
                source_title = row["Source title"]
                volume = row["Volume"]
                pages = f"{row['Page start']}-{row['Page end']}"
                doi = row["DOI"]
                # bibtex_key = generate_bibtex_key(authors, year)
                bibtex_entry = f"@article{{\n" \
                               f"  author = {{{authors}}},\n" \
                               f"  title = {{{title}}},\n" \
                               f"  year = {{{year}}},\n" \
                               f"  number = 1,\n" \
                               f"  journal = {{{source_title}}},\n" \
                               f"  volume = {{{volume}}},\n" \
                               f"  pages = {{{pages}}},\n" \
                               f"  doi = {{{doi}}}\n" \
                               f"}}\n\n"
                bibtexfile.write(bibtex_entry)

# Example usage
convert_csv_to_bibtex("scopus.csv", "scopus.bib")