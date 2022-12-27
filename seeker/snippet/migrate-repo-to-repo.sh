#date: 2022-12-27T16:40:58Z
#url: https://api.github.com/gists/3c9723dac1edbd84ccb893c147fff8a1
#owner: https://api.github.com/users/Kaskyi

export FROM_REPO=my-old-repo
export FROM_DIR=my-traget-dir
export TO_REPO=my-new-repo
export TO_DIR=my-dist-dir

echo Copies the code with history from the folder of the previos repository \
     and pastes into the folder at new repository

MIGRATION_FOLDER=__migration-folder
echo \================================================================
echo Make a $MIGRATION_FOLDER
echo \================================================================
mkdir $MIGRATION_FOLDER
cd $MIGRATION_FOLDER || exit 1

echo \================================================================
echo Clone from $FROM_REPO
echo \================================================================
git clone --progress $FROM_REPO .
git filter-branch --subdirectory-filter $FROM_DIR -- --all

echo \================================================================
echo Move to $TO_DIR
echo \================================================================
mkdir -p $TO_DIR
mv * $TO_DIR

git add .
git commit -m "Extract \"$FROM_DIR\" directory"

echo \================================================================
echo Clone and merge from $TO_REPO
echo \================================================================
MIGRATION_REMOTE=__migration-remote
git remote add $MIGRATION_REMOTE $TO_REPO
git pull $MIGRATION_REMOTE master --allow-unrelated-histories --commit --no-edit


MIGRATION_BRANCH=__migration-branch
echo \================================================================
echo Create a $MIGRATION_BRANCH
echo \================================================================
git checkout -b $MIGRATION_BRANCH
git push -u $MIGRATION_REMOTE $MIGRATION_BRANCH -o merge_request.create