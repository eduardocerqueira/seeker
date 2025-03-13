#date: 2025-03-13T16:43:44Z
#url: https://api.github.com/gists/95601ee32823fa6246561cf3984df0a0
#owner: https://api.github.com/users/avrahamappel-humi

#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p "python3.withPackages (ps: with ps; [ requests pygithub ])"

import requests
from github import Github
from typing import Dict, Tuple
import urllib.parse

# Configuration - Replace these with your actual credentials and project details
LINEAR_API_KEY = "linear-key"
GITHUB_TOKEN = "**********"
LINEAR_TEAM_KEY = "AIOP"  # e.g., "ENG" for ticket IDs like ENG-123
COMPLETED_STATE_NAME = "Done"

# Linear GraphQL endpoint
LINEAR_API_URL = "https://api.linear.app/graphql"

def get_linear_completed_tickets() -> list:
    """Fetch completed tickets assigned to the authenticated user from Linear."""
    query = """
    query {
      user(id: "me") {
        id
        assignedIssues(filter: { state: { name: { eq: "%s" } }, team: { key: { eq: "%s" } } }) {
          nodes {
            id
            identifier
            title
            branchName
            attachments {
              nodes {
                url
                sourceType
              }
            }
          }
        }
      }
    }
    """ % (COMPLETED_STATE_NAME, LINEAR_TEAM_KEY)

    headers = {
        "Authorization": LINEAR_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.post(LINEAR_API_URL, json={"query": query}, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Linear API error: {response.status_code} - {response.text}")
    
    data = response.json()
    return data["data"]["user"]["assignedIssues"]["nodes"]

def extract_pr_details(pr_url: str) -> Tuple[str, int]:
    """Extract repository name and PR number from a GitHub PR URL."""
    parsed_url = urllib.parse.urlparse(pr_url)
    path_parts = parsed_url.path.strip("/").split("/")
    
    if len(path_parts) < 4 or path_parts[2] != "pull":
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
    
    repo_name = f"{path_parts[0]}/{path_parts[1]}"  # e.g., "owner/repo"
    pr_number = int(path_parts[3])  # e.g., 123
    
    return repo_name, pr_number

def extract_pr_url(attachments: list) -> str:
    """Extract GitHub PR URL from Linear ticket attachments."""
    for attachment in attachments:
        if attachment["sourceType"] == "github" and "pull" in attachment["url"]:
            return attachment["url"]
    return None

def get_pr_changes(github_client: Github, pr_url: str) -> Tuple[int, int]:
    """Get the number of additions and deletions from a GitHub pull request."""
    repo_name, pr_number = extract_pr_details(pr_url)
    
    repo = github_client.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    return pr.additions, pr.deletions

def main():
    # Initialize GitHub client
    github_client = "**********"
    
    # Get completed tickets from Linear
    tickets = get_linear_completed_tickets()
    
    total_additions = 0
    total_deletions = 0
    
    print(f"Analyzing {len(tickets)} completed tickets assigned to you...\n")
    
    for ticket in tickets:
        ticket_id = ticket["identifier"]
        ticket_title = ticket["title"]
        pr_url = extract_pr_url(ticket["attachments"]["nodes"])
        
        if not pr_url:
            print(f"Ticket {ticket_id}: '{ticket_title}' - No GitHub PR found")
            continue
        
        try:
            additions, deletions = get_pr_changes(github_client, pr_url)
            total_additions += additions
            total_deletions += deletions
            print(f"Ticket {ticket_id}: '{ticket_title}' - PR: {pr_url}")
            print(f"  Additions: {additions}, Deletions: {deletions}")
        except Exception as e:
            print(f"Ticket {ticket_id}: Error processing PR {pr_url} - {str(e)}")
    
    print("\nSummary:")
    print(f"Total lines added: {total_additions}")
    print(f"Total lines deleted: {total_deletions}")
    print(f"Net change: {total_additions - total_deletions}")

if __name__ == "__main__":
    # Install required packages if not already installed:
    # pip install requests PyGithub
    main()
ain()
