#date: 2021-11-10T17:08:40Z
#url: https://api.github.com/gists/f383716b40d63898ca88be05f4642c94
#owner: https://api.github.com/users/ejbtrd

# Adjust the following variables as necessary
REMOTE=streak
#BRANCH=$(git rev-parse --abbrev-ref HEAD)
BRANCH=twelve
BATCH_SIZE=5000

if [ $1 == "" ]; then
    echo "Specify path to repo first!"
    exit
fi

cd $1

# check if the branch exists on the remote
if git show-ref --quiet --verify refs/remotes/$REMOTE/$BRANCH; then
    # if so, only push the commits that are not on the remote already
    range=$REMOTE/$BRANCH..HEAD
else
    # else push all the commits
    range=HEAD
fi
# count the number of commits to push
n=$(git log --first-parent --format=format:x $range | wc -l)

# push each batch
for i in $(seq $n -$BATCH_SIZE 1); do
    # get the hash of the commit to push
    h=$(git log --first-parent --reverse --format=format:%H --skip $i -n1)
    echo "Pushing $h..."
    git push $REMOTE ${h}:refs/heads/$BRANCH
done
# push the final partial batch
git push $REMOTE HEAD:refs/heads/$BRANCH

cd - &> /dev/null