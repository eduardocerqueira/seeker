#date: 2023-12-07T17:05:31Z
#url: https://api.github.com/gists/b14b1b2e2650f1544881d3e9ab5a1de7
#owner: https://api.github.com/users/moeriki

#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <source_remote> <destination_remote> <pattern1:ref1> [<pattern2:ref2> ...]"
	echo ""
	echo "Examples:"
	echo "  $0 origin github development:master uat 'releases/*'"
	echo ""
	echo "Notes:"
	echo "- All refs should be on the source remote."
	exit 1
fi
#!/bin/bash

# Assign input parameters to variables
SOURCE_REMOTE="$1"
DESTINATION_REMOTE="$2"
shift 2
PATTERNS=("$@")

# Function to perform one-way sync
sync_repository() {
	# Fetch branches and tags from source remote based on the provided patterns
	for pattern in "${PATTERNS[@]}"; do
		IFS=':' read -r -a pattern_parts <<< "$pattern"
		ref_pattern="${pattern_parts[0]}"

		# If destination ref is not provided, use the source ref
		dest_ref="${pattern_parts[1]:-${ref_pattern}}"

		# Fetch refs matching the pattern from the source remote
		for ref in $(git ls-remote --heads $SOURCE_REMOTE $ref_pattern | cut -f2); do
			# Extract branch name
			branch_name=$(basename $ref)

			git fetch $SOURCE_REMOTE $ref
			# Push the remote-tracking branch to the destination remote with the actual branch name
			git push $DESTINATION_REMOTE refs/remotes/$SOURCE_REMOTE/$branch_name:$branch_name
		done

		# Fetch tags matching the pattern from the source remote
		for tag_ref in $(git ls-remote --tags $SOURCE_REMOTE $ref_pattern | cut -f2); do
			# Extract tag name
			tag_name=$(basename $tag_ref)

			git fetch $SOURCE_REMOTE $tag_ref
			# Push the remote-tracking tag to the destination remote with the actual tag name
			git push $DESTINATION_REMOTE refs/remotes/$SOURCE_REMOTE/$tag_name:$tag_name
		done
	done
}

# Call the sync_repository function
sync_repository

# Exit script
exit 0
