#date: 2024-10-31T16:48:59Z
#url: https://api.github.com/gists/8e71b2676f694b42f356d93d179f3a75
#owner: https://api.github.com/users/thor314

# pip install PyGithub requests
import os
import sys
import requests
from github import Github
from datetime import datetime

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"r "**********"a "**********"n "**********"s "**********"f "**********"e "**********"r "**********"_ "**********"r "**********"e "**********"p "**********"o "**********"s "**********"( "**********"s "**********"o "**********"u "**********"r "**********"c "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"r "**********"e "**********"p "**********"o "**********"s "**********", "**********"  "**********"t "**********"a "**********"r "**********"g "**********"e "**********"t "**********"_ "**********"o "**********"r "**********"g "**********") "**********": "**********"
    """
    Transfer multiple GitHub repositories to a target organization.
    
    Args:
        source_token (str): "**********":org scope
        repos (list): List of repository names to transfer
        target_org (str): Name of the target organization
    """
    # Initialize GitHub client
    g = "**********"
    
    # Log file setup
    log_file = f"repo_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    for repo_name in repos:
        try:
            # Get repository object
            repo = g.get_user().get_repo(repo_name)
            
            print(f"Starting transfer of {repo_name} to {target_org}...")
            
            # Initiate transfer
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": "**********"
            }
            
            transfer_url = f"https://api.github.com/repos/{repo.full_name}/transfer"
            transfer_data = {"new_owner": target_org}
            
            response = requests.post(transfer_url, headers=headers, json=transfer_data)
            
            if response.status_code in [202, 200]:
                print(f"✓ Successfully initiated transfer of {repo_name}")
                with open(log_file, 'a') as f:
                    f.write(f"SUCCESS: {repo_name} transfer initiated to {target_org}\n")
            else:
                print(f"✗ Failed to transfer {repo_name}: {response.json().get('message')}")
                with open(log_file, 'a') as f:
                    f.write(f"FAILED: {repo_name} - {response.json().get('message')}\n")
                    
        except Exception as e:
            print(f"✗ Error processing {repo_name}: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {repo_name} - {str(e)}\n")

def main():
    # Check for GitHub token in environment
    token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        print("Error: "**********"
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py TARGET_ORG REPO1 [REPO2 REPO3 ...]")
        print("Example: python script.py new-org repo1 repo2 repo3")
        sys.exit(1)
    
    # Get target org and repos from command line arguments
    target_org = sys.argv[1]
    repos = sys.argv[2:]
    
    print(f"Preparing to transfer {len(repos)} repositories to {target_org}")
    transfer_repos(token, repos, target_org)

if __name__ == "__main__":
    main()
