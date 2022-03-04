#date: 2022-03-04T17:13:43Z
#url: https://api.github.com/gists/1929d0acf37d29c0c41a078917b1ef3b
#owner: https://api.github.com/users/sdtblck

# Create empty branch.
git checkout --orphan review
git rm -rf .
git commit --allow-empty -m "Create empty branch"
git push --set-upstream origin review

# Create `project` branch from `master` current state.
git checkout -b project
git merge master --allow-unrelated-histories
git push --set-upstream origin project

# Open a Pull-Request on the Github repository from `project` to `review`. Then you can perform a full-code review.
