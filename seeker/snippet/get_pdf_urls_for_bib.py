#date: 2023-05-17T17:08:47Z
#url: https://api.github.com/gists/40a1dbf70ed9d16c1a2dfc4230e2f3e8
#owner: https://api.github.com/users/atrettin

"""
A simple script to get the PDF URLs for all entries in a given BibTeX file and save them to a new BibTeX file.
"""
import bibtexparser
from requests.exceptions import HTTPError
from unpywall import Unpywall
from unpywall.utils import UnpywallCredentials


def get_pdf_url(doi):
    pdf_url = "nan"
    unpywall_df = None
    try:
        unpywall_df = Unpywall.doi(dois=[doi])
    except HTTPError:
        print(f"DOI {doi} not found in Unpaywall database.")
    if (
        unpywall_df is not None
        and "best_oa_location.url_for_pdf" in unpywall_df.columns
    ):
        pdf_url = unpywall_df["best_oa_location.url_for_pdf"][0]
    return pdf_url if pdf_url is not None else "nan"


def main(args):
    UnpywallCredentials(args.email)
    # Now that we have all the PDF URLs, we can also update the bibliography file to fill in the missing PDF links.
    bib_file = args.input_file
    bib_data = bibtexparser.load(open(bib_file))

    for i, entry in enumerate(bib_data.entries):
        # First, find the matching entry in the dataframe (match DOI)
        try:
            doi = str(entry["doi"])
        except KeyError:
            doi = ""
        if doi == "":
            continue
        # Strip "https://doi.org/" from the DOI numbers if present
        doi = doi.replace("https://doi.org/", "")
        # Strip "http://dx.doi.org/" from the DOI numbers if present
        doi = doi.replace("http://dx.doi.org/", "")
        pdf_url = get_pdf_url(doi)
        if pdf_url == "nan" or pdf_url is None:
            continue
        # Now, update the PDF URL. The field name of the bibtex entry is "url"
        bib_data.entries[i]["url"] = pdf_url

    # Finally, save the updated bibliography file.
    with open(args.output_file, "w") as bibtex_file:
        bibtexparser.dump(bib_data, bibtex_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get PDF URLs for all entries in a given BibTeX file and save them to a new BibTeX file."
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="The input BibTeX file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="The output BibTeX file.",
    )
    parser.add_argument(
        "-e",
        "--email",
        type=str,
        required=True,
        help="Your email address (required by Unpaywall).",
    )
    args = parser.parse_args()
    main(args)
