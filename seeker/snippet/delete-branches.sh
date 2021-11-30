#date: 2021-11-30T17:12:06Z
#url: https://api.github.com/gists/c435a3bc27bd09620498ef5eb54bbfa7
#owner: https://api.github.com/users/marchuffnagle

# List local branches which have already been merged into the `develop` branch

git branch --merged develop |\
  grep -Ev '(^\*|develop|main|master)'

# Delete local branches which have already been merged into the `develop` branch

git branch --merged develop |\
  grep -Ev '(^\*|develop|main|master)' |\
  xargs git branch -d
