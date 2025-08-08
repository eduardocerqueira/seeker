#date: 2025-08-08T16:55:30Z
#url: https://api.github.com/gists/5cec7359efc83db22aa54cae7c377b00
#owner: https://api.github.com/users/TheAlchemistNerd

# pip install python-dotenv pandas requests
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# === LOAD ENV VARIABLES ===
load_dotenv()
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = "**********"

if not GITHUB_USERNAME:
    raise ValueError("❌ GITHUB_USERNAME not found in .env file.")
 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"G "**********"I "**********"T "**********"H "**********"U "**********"B "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
    print("⚠️ No GITHUB_TOKEN found — you may hit API rate limits for large repos.")

# === API REQUEST ===
url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
headers = {"Accept": "application/vnd.github.v3+json"}

 "**********"i "**********"f "**********"  "**********"G "**********"I "**********"T "**********"H "**********"U "**********"B "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
    response = "**********"=headers, auth=(GITHUB_USERNAME, GITHUB_TOKEN))
else:
    response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise Exception(f"GitHub API error: {response.status_code} - {response.text}")

repos = response.json()

# === EXTRACT DATA ===
data = []
for repo in repos:
    name = repo["name"]
    description = repo["description"] or "No description provided"
    language = repo["language"] or "Not specified"
    stars = repo["stargazers_count"]
    forks = repo["forks_count"]

    # Get commit count for each repo
    commits_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{name}/commits"
    commits_resp = "**********"=headers, auth=(GITHUB_USERNAME, GITHUB_TOKEN))
    commit_count = len(commits_resp.json()) if commits_resp.status_code == 200 else "N/A"

    data.append([name, description, language, stars, forks, commit_count])

# === FORMAT INTO DATAFRAME ===
df = pd.DataFrame(data, columns=["Name", "Description", "Language", "Stars", "Forks", "Commits"])

# === OUTPUT ===
print("\n📌 GitHub Projects Summary:\n")
print(df.to_string(index=False))

# === OPTIONAL: SAVE TO CSV ===
df.to_csv("github_projects_summary.csv", index=False)
print("\n✅ Saved to github_projects_summary.csv")
