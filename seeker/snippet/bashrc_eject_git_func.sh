#date: 2024-05-17T17:02:42Z
#url: https://api.github.com/gists/19cfcbcc117dc116afb7a7b7a171249e
#owner: https://api.github.com/users/lmmx

function ejectit () {
        # e: exit immediately on error; u: treat unset variables as an error
        set -eu
        # Create a new PR with auto-filled content (title from branch name, commits as body)
        gh pr create --fill
        # Merge the new PR (squashing commits) automatically [after checks pass], then delete the branch
        gh pr merge --squash --auto --delete-branch
        # Get an absolute path to the repo (top level directory)
        local REPO=$(git rev-parse --show-toplevel)
        # Double check that the .git/ directory is under it (be careful not to get the wrong target)
        if [[ -d $REPO/.git ]]; then
                # cd into the repo's parent directory and then delete it (as if 'ejecting' the repo)
                cd $REPO/.. && rm -rf $REPO;
        else
                echo "Not deleting the local repo directory: $REPO/.git was not found"
        fi
}