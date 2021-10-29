#date: 2021-10-29T17:03:38Z
#url: https://api.github.com/gists/f6271ff22b4ed1f221b0f57ef12b0dfe
#owner: https://api.github.com/users/dulkith

if [[ "$#" -ne 1 ]]; then
	echo "You must specify a commit message as an argument."
	exit 1
fi

COMMIT_MESSAGE="$1"
git submodule foreach "echo 'Adding all files.'; git add ."
git submodule foreach "echo 'Stashing.'; git stash"
git submodule foreach "echo 'Checking out master branch.'; git checkout master"
git submodule foreach "echo 'Applying stash.'; git stash apply"
git submodule foreach "echo 'Readding all files.'; git add ."
git submodule foreach "echo 'Committing changes.'; git commit -a -q -m '${COMMIT_MESSAGE}' || :"
echo 'Adding main repo changes.'
git add .
echo 'Committing changes to main repo.'
git commit -a -m "${COMMIT_MESSAGE}"