#date: 2025-09-25T16:43:31Z
#url: https://api.github.com/gists/86df583a07abe8c750ee8cb15617c2da
#owner: https://api.github.com/users/r2g

## Pre-requisite: You have to know your last commit message from your deleted branch.
git reflog
# Search for message in the list
# a901eda HEAD@{18}: commit: <last commit message>

# Now you have two options, either checkout revision or HEAD
git checkout a901eda 
# Or
git checkout HEAD@{18}

# Create branch
git branch recovered-branch

# You may want to push that back to remote
git push origin recovered-branch:recovered-branch