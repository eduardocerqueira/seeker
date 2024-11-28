#date: 2024-11-28T17:09:16Z
#url: https://api.github.com/gists/679b5600f1d1d0a25c16579c426ef065
#owner: https://api.github.com/users/rubensa

# The command finds the most recent tag that is reachable from a commit.
# If the tag points to the commit, then only the tag is shown.
# Otherwise, it suffixes the tag name with the number of additional commits on top of the tagged object 
# and the abbreviated object name of the most recent commit.
git describe

# With --abbrev set to 0, the command can be used to find the closest tagname without any suffix:
git describe --abbrev=0

# other examples
git describe --abbrev=0 --tags # gets tag from current branch
git describe --tags `git rev-list --tags --max-count=1` # gets tags across all branches, not just the current branch


