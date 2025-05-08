#date: 2025-05-08T16:39:16Z
#url: https://api.github.com/gists/78a2351eb54b323deb876cb4c5c5a15e
#owner: https://api.github.com/users/mateusoliveira43

# in parent branch, grab commit hash
git log
git checkout <release-1.4>
git checkout -b <cp-1.4-branch-name>
git cherry-pick -x <commit-hash>
# resolve conflicts, if necessary
# git add
# git cherry-pick --continue
git push origin <cp-1.4-branch-name>