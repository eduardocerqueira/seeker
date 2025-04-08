#date: 2025-04-08T17:08:46Z
#url: https://api.github.com/gists/5f1a6d7eb0560b9892ae4a7ce02947aa
#owner: https://api.github.com/users/IlanVivanco

#!/bin/bash

# Step 1: Generate commit log (oldest first) with parent info
git log --pretty=format:"%h|%ad|%an|%s|%p" --date=short --reverse >git_log.txt

# Step 2: Initialize
declare -A seen
>rebase-plan.txt

# Step 3: Process each line
while IFS='|' read -r hash date author subject parents; do
	# Count number of parents to detect merge commits
	parent_count=$(echo $parents | wc -w)

	# Skip merge commits (commits with more than one parent)
	if [ $parent_count -gt 1 ]; then
		echo "# Skipping merge commit: $hash $subject" >>rebase-plan.txt
		continue
	fi

	key="$date|$author"
	if [[ -z "${seen[$key]}" ]]; then
		echo "pick $hash $subject" >>rebase-plan.txt
		seen[$key]=1
	else
		echo "fixup $hash $subject" >>rebase-plan.txt
	fi
done <git_log.txt

# Create a helper function for auto-rebase
auto_rebase() {
	echo "⚙️ Starting auto-rebase with conflict resolution..."
	GIT_EDITOR="cat rebase-plan.txt >" git rebase -i --root --strategy-option=theirs
	echo "✅ Auto-rebase completed!"
}

echo "✅ Rebase plan written to rebase-plan.txt"
echo ""
echo "Choose an option:"
echo "1) Manual rebase (you'll resolve conflicts)"
echo "   👉 Run: git rebase -i --root, then replace contents with rebase-plan.txt"
echo ""
echo "2) Auto-rebase (automatically accept incoming changes for conflicts)"
echo "   👉 Run: ./rebase.sh --auto"
echo ""
echo "📝 If you encounter conflicts during manual rebase:"
echo "  1. Resolve conflicts in each file"
echo "  2. git add <resolved-files>"
echo "  3. git rebase --continue"
echo ""
echo "   To skip a problematic commit: git rebase --skip"
echo "   To abort the rebase: git rebase --abort"

# Check if auto-rebase was requested
if [[ "$1" == "--auto" ]]; then
	auto_rebase
fi
