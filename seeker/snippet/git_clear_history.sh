#date: 2022-04-13T17:04:37Z
#url: https://api.github.com/gists/ff0c4045599afc379871719861571d3d
#owner: https://api.github.com/users/bapti

-- Remove the history from 
rm -rf .git

-- recreate the repos from the current content only
git init
git add .
git commit -m "Initial commit"

-- push to the github remote repos ensuring you overwrite history
git remote add origin git@github.com:<YOUR ACCOUNT>/<YOUR REPOS>.git
git push -u --force origin master