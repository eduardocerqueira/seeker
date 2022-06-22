#date: 2022-06-22T17:10:46Z
#url: https://api.github.com/gists/2cbb45dadb091c075f5e105f38afcbc7
#owner: https://api.github.com/users/ashamin-tr

#!/bin/bash

# Dependencies: github-cli, git

function failed() {
	echo $1
  echo "Failed. Exiting..."
	exit
}


# variables ---------------------------------------------------------------------------------------
base_branch=""
target_branch=""
pull=false
push=false
force=false
remote=`git remote`
username=""
help=false
pr=""
labels=""

# arguments-----------------------------------------------------------------------------------------
function help()
{
  echo "-t|--target     a target branch"
  echo "-b|--base       a base branch"
  echo "-u|--user       username for a branch name"
  echo "--pull          try to pull changes from server"
  echo "--remote        remote name"
  echo "-f|--force      force remove existing merge branch"
  echo "--push          push result of merge to server"
  echo "--pr            create pr arg: github"
  echo "--labels        add labels to pr"
}

while [ -n "$1" ]; do 
	case "$1" in 	
		(-b | --base) base_branch="$2" ;;
		(-t | --target) target_branch="$2" ;;
		(-u | --user) username="$2" ;;
		(--pull) pull=true ;;
    (--push) push=true ;;
    (--remote) remote="$2" ;;
    (-f | --force) force=true ;;
    (--pr) pr="$2" ;;
    (-l | --labels) labels="$2" ;;
		(-h | --help) help; exit ;;
	esac
	shift	
done

[ -z $base_branch ] && failed "Argument -b | --base required"
[ -z $target_branch ] && failed "Argument -t | --target required"


# script -------------------------------------------------------------------------------------------
merge_branch="$base_branch/integrate/$target_branch"

[ -z $username ] || merge_branch="$username/$merge_branch"

out_str="Merge '$target_branch' into '$base_branch' through '$merge_branch'"
[ $push = true ] && out_str="$out_str and push to '$remote'"

echo "$out_str"
[ $pull = true ] && echo "Try to pull changes from server for branches '$base_branch' and '$target_branch'"

git fetch --all

git checkout $base_branch || failed "Can't checkout to '$base_branch'"

if $pull = true
then
  echo "Try to pull to '$base_branch'"
  git pull

  echo "Try to pull to '$target_branch'"
  git checkout $target_branch || failed "Can't checkout to '$target_branch'"

  git pull
  git checkout $base_branch || failed "Can't checkout to '$base_branch'"
fi

[ $force = true ] && git branch -D $merge_branch
git checkout -b $merge_branch || failed "Can't create branch '$merge_branch'"

git merge --no-edit $target_branch || failed "Can't merge '$target_branch' into '$merge_branch'"

if $push = true 
then
  git push -u $remote $merge_branch || failed "Can't push '$merge_branch' to '$remote'"

  if [ $pr = "github" ]
  then
    gh auth status || failed "Please login 'gh' (github-cli) before executing the script"
    
    gh pr create --base $base_branch --label "$labels" --title "Integrate '$target_branch' to '$base_branch'" --body "" --web
  fi

fi


