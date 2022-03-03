#date: 2022-03-03T16:54:20Z
#url: https://api.github.com/gists/18db9c4e9b1fe7d4a658ddfcbb9bcb79
#owner: https://api.github.com/users/LaurierRoy

# If you work with git, you've probably had that nagging sensation of not knowing what branch you are on. Worry no longer!

export PS1="\\w:\$(git branch 2>/dev/null | grep '^*' | colrm 1 2)\$ "

# This will change your prompt to display not only your working directory but also your current git branch, if you have one. Pretty nifty!

# ~/code/web:beta_directory$ git checkout master
# Switched to branch "master"
# ~/code/web:master$ git checkout beta_directory
# Switched to branch "beta_directory"
# ~/code/web:beta_directory$ 