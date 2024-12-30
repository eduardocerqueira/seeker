#date: 2024-12-30T16:30:20Z
#url: https://api.github.com/gists/ebe37f2105da2a83781d10cc73078592
#owner: https://api.github.com/users/Rajeshcn02

import os
import requests
import csv
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename="github_actions_details.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_repositories(api_url, headers):
    repositories = []
    page = 1

    while True:
        url = f"{api_url}/orgs/{ORG_NAME}/repos?per_page=100&page={page}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(f"Error fetching repositories: {response.status_code} - {response.text}")
            break

        repos = response.json()
        if not repos:
            break

        repositories.extend(repos)
        page += 1

    return repositories


def fetch_actions_details(api_url, headers, repo_name):
    # Check if Actions is enabled
    actions_url = f"{api_url}/repos/{ORG_NAME}/{repo_name}/actions/permissions"
    response = requests.get(actions_url, headers=headers)

    if response.status_code != 200:
        logging.error(f"Error fetching Actions details for {repo_name}: {response.status_code} - {response.text}")
        return None, None, None

    actions_data = response.json()
    is_enabled = actions_data.get("enabled", False)

    # Fetch Runners
    runners_url = f"{api_url}/repos/{ORG_NAME}/{repo_name}/actions/runners"
    runners_response = requests.get(runners_url, headers=headers)
    runners = runners_response.json().get("runners", []) if runners_response.status_code == 200 else []

    # Fetch Secrets
    secrets_url = "**********"
    secrets_response = "**********"=headers)
    secrets = "**********"== 200 else []

    return is_enabled, runners, secrets


def save_to_csv(repositories, filename="github_actions_details.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Repository Name", "Actions Enabled", "Runners", "Secrets"
        ])

        for repo in repositories:
            name = repo["name"]

            actions_enabled, runners, secrets = "**********"

            # Format runners and secrets
            runners_list = [f"{runner['name']} ({runner['os']})" for runner in runners]
            secrets_list = "**********"

            writer.writerow([
                name, actions_enabled, ", ".join(runners_list), ", ".join(secrets_list)
            ])


def main():
    load_dotenv()

    global GITHUB_TOKEN, API_URL, ORG_NAME, HEADERS
    GITHUB_TOKEN = "**********"
    API_URL = os.getenv("GITHUB_SERVER_URL")  # Example: https://github.ecanarys.com/api/v3
    ORG_NAME = os.getenv("GITHUB_ORG")

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"G "**********"I "**********"T "**********"H "**********"U "**********"B "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"A "**********"P "**********"I "**********"_ "**********"U "**********"R "**********"L "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"O "**********"R "**********"G "**********"_ "**********"N "**********"A "**********"M "**********"E "**********": "**********"
        logging.error("Error: "**********"
        return

    HEADERS = {"Authorization": "**********"

    logging.info("Fetching repositories...")
    repositories = fetch_repositories(API_URL, HEADERS)

    if not repositories:
        logging.error("No repositories found or an error occurred.")
        return

    logging.info("Saving GitHub Actions details to CSV...")
    save_to_csv(repositories)
    logging.info("GitHub Actions details saved to 'github_actions_details.csv'.")

if __name__ == "__main__":
    main()
