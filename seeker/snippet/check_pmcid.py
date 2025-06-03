#date: 2025-06-03T17:10:10Z
#url: https://api.github.com/gists/ac6db1889b6b64953166de64eb8a554e
#owner: https://api.github.com/users/cannin

import requests
from diskcache import Cache
import os

# Ensure cache directory exists
os.makedirs('cache', exist_ok=True)
cache = Cache('cache')


def check_pmcid(pmid, email="augustin@nih.gov"):
    """Check if a given PMID has a corresponding PMCID using NCBI's ID Converter API.

    Queries the NCBI ID Converter API to find if a PubMed ID has a corresponding
    PubMed Central ID (PMCID).

    Parameters
    ----------
    pmid : str
        The PubMed ID to check.

    Returns
    -------
    str or None
        The corresponding PMCID if available, or None if no PMCID exists.
    """
    key = f"pmcid_{pmid}"
    if key in cache:
        return cache[key]

    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmid,
        "format": "json",
        "tool": "return_pmcid",
        "email": email
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "records" in data and data["records"]:
            record = data["records"][0]
            if "pmcid" in record:
                cache[key] = record['pmcid']
                return record['pmcid']
            else:
                cache[key] = None
                return None
        else:
            cache[key] = None
            return None

    except requests.exceptions.RequestException:
        # Log the error if needed, but don't expose the exception
        print(f"Error looking up PMCID for {pmid}")
        cache[key] = None
        return None


# Example usage:
if __name__ == "__main__":
    pmid_input = input("Enter a PMID: ").strip()
    result = check_pmcid(pmid_input)
    print(result or "No PMCID found")
