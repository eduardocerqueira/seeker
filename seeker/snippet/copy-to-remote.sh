#date: 2024-06-19T17:12:35Z
#url: https://api.github.com/gists/d40dbcc7a39cb3aa2628d8710b5bcb32
#owner: https://api.github.com/users/WatweA

#!/usr/bin/env bash
set -euo pipefail

# CREDIT: https://stackoverflow.com/a/51468389
# ADAPTED FROM ABOVE ANSWER, ORIGINAL AUTHORS:
# - Guillaume Jacquenot
# - Daniel Harding

# Adjust the following variables as necessary
REMOTE=${1}
echo "Using remote: '$REMOTE'"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BATCH_SIZE=1000

# check if the branch exists on the remote
if git show-ref --quiet --verify refs/remotes/$REMOTE/$BRANCH; then
    # if so, only push the commits that are not on the remote already
    range=$REMOTE/$BRANCH..HEAD
    echo "Brach '$BRANCH' exists on remote '$REMOTE', only pushing $range"
else
    # else push all the commits
    range=HEAD
    echo "Brach '$BRANCH' not found on remote '$REMOTE', pushing all commits until $range"
fi
# count the number of commits to push
n=$(git log --first-parent --format=format:x $range | wc -l)
echo "Pushing $n total commits in batches of $BATCH_SIZE commits"

# push each batch
for i in $(seq $n -$BATCH_SIZE 1); do
    # get the hash of the commit to push
    h=$(git log --first-parent --reverse --format=format:%H --skip $i -n1)
    echo "Pushing $h..."
    git push $REMOTE ${h}:refs/heads/$BRANCH
done
# push the final partial batch
git push $REMOTE HEAD:refs/heads/$BRANCH
