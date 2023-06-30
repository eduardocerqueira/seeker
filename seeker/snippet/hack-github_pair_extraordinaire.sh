#date: 2023-06-30T16:34:56Z
#url: https://api.github.com/gists/63953c6821d5d557f4360c3e16e20160
#owner: https://api.github.com/users/th2ch-g

#!/bin/bash
set -e

author_name_of_repo=""
name_of_repo=""
git_remote_add_origin=""

USAGE='
PROGRAM:
    ./hack-github_pair_extraordinaire.sh

REQUIREMENT:
    gh: github-CLI tools, install by "brew install gh" and make sure that you have already logined

USAGE:
    ./hack-github_pair_extraordinaire.sh [number_of_merge] [emails]

EXAMPLE:
    ./hack-github_pair_extraordinaire.sh -h                                                     show help message
    ./hack-github_pair_extraordinaire.sh 1024 hoge1@hoge.com hoge2@hoge.com                     make git-commit and merge 1024 times using given 2 emails
    ./hack-github_pair_extraordinaire.sh 24 hoge1@hoge.com hoge2@hoge.com hoge3@hoge.com        make git-commit and merge 24 times using given 3 emails
'

# help message
if [[ $1 = "-h" ]] || [[ $1 = "--help" ]] || [[ $1 = "-help" ]] || [[ -z $1 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

# paramter check
if [[ -z ${author_name_of_repo} ]] || [[ -z ${name_of_repo} ]] || [[ -z ${git_remote_add_origin} ]]; then
    echo "input paramters in this script" >&2
    exit 1
fi

# make git-commit message
message='add

'
number=0
for email in "$@";
do
    number=$(( $number + 1 ))
    [ $number -eq 1 ] && continue
message="${message}Co-authored-by: someone <$email>
"
done
echo "[INFO] git-commit message: $message" >&1

# make directory
pid="$$"
mkdir tmp-${pid}
cd tmp-${pid}
git init --initial-branch main
git remote add origin ${git_remote_add_origin}
touch initialize.txt
git add -A
git commit -m "init"
git push --set-upstream origin main
export GH_DEFAULT_REPO="${author_name_of_repo}/${name_of_repo}"
for num in $(seq $1);
do
    echo "[INFO] ${num}th start" >&1
    touch empty_${num}.txt
    git add -A
    git checkout -b empty_${num}
    git commit -m "$message"
    git push --set-upstream origin empty_${num}
    gh pr create \
        --base main --title "add" --body "" --repo "${author_name_of_repo}/${name_of_repo}"
    gh pr merge -m -d
    sleep 3
done
cd ..

echo ""
echo ""
echo ""
echo ""
echo "all done" >&1