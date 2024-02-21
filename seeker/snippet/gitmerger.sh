#date: 2024-02-21T17:01:24Z
#url: https://api.github.com/gists/4152734eb1e1befa699c354008aa565c
#owner: https://api.github.com/users/antoninadert

# HOW TO USE :
# When trying to "git pull" and you get a conflict (as per git output), run `sh gitter.sh` in your terminal to handle conflict and help finishing merge.
# This script works with a versioning strategy where everyone works on the same branch 'main'. Adapt to your needs.
# Once merge conflicts are handled, you can `git pull` then `git push`

# If the bash script doesn't work, make it executable. In bash execute : `chmod +x gitmerger.sh` (it will update access rights)
# Tested on MacOS, should work on Linux too.  


git fetch origin main
# Run git merge in a subshell and capture output
merge_output=$( (git merge origin/main) 2>&1 )
if [[ "$merge_output" == *"error: Your local changes to the following files would be overwritten by merge:"* ]]; then
  echo "Staging and Committing to prepare conflict resolution"
  git add .
  git commit -am "Preparing conflict resolution"
  merge_output=$( (git merge origin/main) 2>&1 )
fi

if [[ "$merge_output" == *"error: Your local changes to the following files would be overwritten by merge:"* || "$merge_output" == *"error: Merging is not possible because you have unmerged files."* || "$merge_output" == *"CONFLICT (content): Merge conflict in"* ]]; then
  echo "You need to solve conflicts manually. Opening VS code..."
  # List conflicted files and open them in VSCode if possible
  for conflicted_file in $(git diff --name-only --diff-filter=U); do
    echo "Opening conflicted file: $conflicted_file"
    code "$conflicted_file"
  done
  # Prompt user for choice
  read -p "Please save files in conflict. When conflicts are solved, enter "1" to merge. Enter "2" to abort conflict resolution): " user_choice
  # Process user choice
  case $user_choice in
    1)
      echo "Conflict resolved manually. Let's merge..."
      git add .
      git commit -am "Conflict resolved manually. Ready to merge"
      git merge origin/main
      ;;
    2)
      echo "Aborting conflict resolution."
      exit 1
      ;;
    *)
      echo "Invalid choice. Please press 1 to proceed with merge or 2 to abort."
      ;;
  esac

  else
    git merge origin/main
fi