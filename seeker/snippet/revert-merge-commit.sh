#date: 2024-01-12T16:49:17Z
#url: https://api.github.com/gists/023ce899898d47e1be6450d1614223a4
#owner: https://api.github.com/users/wlizama

# ensure you are on the correct branch where the merge commit needs to be undone
git checkout main

# find the hash of the merge commit that you want to undo
git log

# Use git revert with the -m option followed by the commit hash to revert the 
# merge commit. The -m option specifies the mainline parent, which is usually the branch you merged into
git revert -m 1 <merge_commit_hash>

# push the changes to the remote 
git push