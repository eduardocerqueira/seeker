#date: 2024-10-31T16:50:44Z
#url: https://api.github.com/gists/70905e5d9dcc829ae49aab49e85954af
#owner: https://api.github.com/users/bittremieux

import json
import time

import requests
import tqdm
from metapub import PubMedFetcher


fetcher = PubMedFetcher(email="YOURNAME@MAIL.COM")

GITHUB_TOKEN = "**********"
GITHUB_HEADERS = {
    "Authorization": "**********"
}


def extract_github_links(article):
    """Extract GitHub URLs from article XML content."""
    article_text = article.abstract if article.abstract else ""
    article_text += (
        " " + article.full_text
        if hasattr(article, "full_text") and article.full_text
        else ""
    )
    urls = []
    for word in article_text.split():
        if "github.com" in word:
            url = word.strip(".,()")
            for suffix in [".git", "releases", "releases/"]:
                url = url.removesuffix(suffix)
            urls.append(url)
    return urls


def get_github_repo_info(repo_url):
    """Collect GitHub repository statistics."""
    repo_info = {}
    repo_name = "/".join(repo_url.split("/")[-2:])

    repo_details_url = f"https://api.github.com/repos/{repo_name}"
    repo_response = requests.get(repo_details_url, headers=GITHUB_HEADERS)
    if repo_response.status_code == 200:
        repo_data = repo_response.json()
        repo_info["stars"] = repo_data.get("stargazers_count")
        license = l if (l := repo_data.get("license", {})) is not None else {}
        repo_info["license"] = license.get("spdx_id")

        contributors_url = f"https://api.github.com/repos/{repo_name}/contributors"
        contributors_response = requests.get(contributors_url, headers=GITHUB_HEADERS)
        if contributors_response.status_code == 200:
            repo_info["contributors"] = len(contributors_response.json())
    else:
        print(f"Failed to retrieve GitHub data for {repo_name}")
    return repo_info


def get_citations(paper_doi):
    """Get number of paper citations from CorssRef."""
    crossref_url = f"https://api.crossref.org/works/{paper_doi}"
    response = requests.get(crossref_url)
    if response.status_code == 200:
        return response.json().get("message", {}).get("is-referenced-by-count", 0)
    else:
        return None


if __name__ == "__main__":
    results = []
    query = '("Journal of Proteome Research"[Journal]) AND (GitHub[Text Word])'
    for pmid in tqdm.tqdm(fetcher.pmids_for_query(query)):
        article = fetcher.article_by_pmid(pmid)
        if article.doi:
            for github_url in extract_github_links(article):
                paper_info = {
                    "doi": article.doi,
                    "github_url": github_url,
                    "citations": get_citations(article.doi),
                }
                paper_info.update(get_github_repo_info(paper_info["github_url"]))
                results.append(paper_info)

                # To avoid rate limiting on APIs.
                time.sleep(1)

    with open("jpr_oss_license.json", "w") as f:
        json.dump(results, f)json.dump(results, f)