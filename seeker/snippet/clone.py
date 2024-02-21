#date: 2024-02-21T17:00:51Z
#url: https://api.github.com/gists/061013d41085b370ea0268af145759a8
#owner: https://api.github.com/users/mvandermeulen

from github import Github  # pip install PyGithub
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Clone Github Repositories')
    parser.add_argument('-u', '--username', dest='github_username', required=True, help='Enter a Github username')
    parser.add_argument('-f', '--folder', dest='folder_name', default='', help='Enter a folder to store repositories')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values using the arguments' attributes
    username = args.github_username
    clone_directory = args.folder_name if args.folder_name else username

    # Authenticate with the GitHub API using a personal access token
    # You can generate a personal access token with the "repo" scope at https: "**********"
    access_token = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        print('Github access token is not set')
        return None
    
    g = "**********"

    # Get the user object
    user = g.get_user(username)
    total_repos = user.get_repos().totalCount

    # Iterate through each repository owned by the user and clone it
    count = 1
    for repo in user.get_repos():
        print(f'Repo {count} / {total_repos}') 
        count += 1
        repo_directory = os.path.join(clone_directory, repo.name)

        if os.path.exists(repo_directory):
            # If the repository directory already exists, pull the latest code
            print(f"Pulling latest code for {repo.name}")
            try:
                os.system(f"cd {repo_directory} && git pull")
                # to pull all branches from origin
                #os.system(f"cd {repo_directory} && git pull --all")
            except Exception as error:
                print(
                    f"Error while trying to pull lastest code from user: {username} - repo: {repo.name} - Error: {error}"
                )

        else:
            # If the repository directory doesn't exist, clone the repository
            print(f"Cloning {repo.name}")
            os.system(f"git clone {repo.clone_url} {repo_directory}")


if __name__ == '__main__':
    main()
