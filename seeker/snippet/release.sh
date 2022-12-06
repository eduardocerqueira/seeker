#date: 2022-12-06T16:47:09Z
#url: https://api.github.com/gists/13e3979308c5f5d9454a0c1a2fbb7cb0
#owner: https://api.github.com/users/spenserblack

#!/bin/bash
# Generates a release tag from RELEASE_NOTES, temporarily setting
# comments to ; to permit Markdown, creates the tag, then
# clears RELEASE_NOTES and resets git comments.
set -e

TAG_NAME=$1
TAG_TARGET=$2

if git config core.commentChar; then
	HAD_COMMENT_CHAR=1
	COMMENT_CHAR=$(git config --get core.commentChar)
else
	HAD_COMMENT_CHAR=0
fi

git config core.commentChar ';'
git tag -F RELEASE_NOTES $TAG_NAME $TAG_TARGET

# Revert to original state
if [ $HAD_COMMENT_CHAR -eq 1 ]; then
	git config core.commentChar $COMMENT_CHAR
else
	git config --unset core.commentChar
fi

echo > RELEASE_NOTES
git add RELEASE_NOTES
git commit -m "Reset RELEASE_NOTES"
