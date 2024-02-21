#date: 2024-02-21T17:00:52Z
#url: https://api.github.com/gists/4e391d03534fbe295fcc88ef57dd65a5
#owner: https://api.github.com/users/mvandermeulen

#  REFER: https://stackoverflow.com/questions/63427607/python-upload-files-directly-to-github-using-pygithub
# Needed Packages
#   PyGithub
#   argparse
#################
# PS: This will only create new file ... does not update the existing file.
#################


import sys
import argparse
from github import Github


GIT_TOKEN= "**********"
parser = argparse.ArgumentParser()


parser.add_argument("-r", "--Repo", help = "Repo name")
parser.add_argument("-f", "--File", help = "File")
parser.add_argument("-p", "--Path", help = "Path")
parser.add_argument("-b", "--Branch", help = "Branch", default="main")
parser.add_argument("-c", "--Commit", help = "Commit", default="committing to main")


args = parser.parse_args()

file_location = args.File
repo_name = args.Repo
path = args.Path
commit_message = args.Commit
branch = args.Branch


#github login
g = "**********"

# get repo
repo = g.get_repo(repo_name)

# get contents
file_content=None
with open(file=file_location, mode="r") as target_file_content:
    file_content = target_file_content.read()

print(file_content)
    
#create new branch if doesnt exists
try:
    repo.get_branch(branch)
except:
    sb = repo.get_branch("main")
    #Create new branch
    repo.create_git_ref(ref="refs/heads/"+f"{branch}", sha= sb.commit.sha)


# create file
repo.create_file(path=path, message=commit_message, content=file_content, branch=branch)

# create PR
body = f'''
ADDING THE FILE {file_location} 
'''
repo.create_pull(title=f"Adding {path}", body=body, head=branch, base="main")dy=body, head=branch, base="main")