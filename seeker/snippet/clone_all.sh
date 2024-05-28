#date: 2024-05-28T16:51:12Z
#url: https://api.github.com/gists/97cb951462b310ac3054b532ccb82282
#owner: https://api.github.com/users/thelooter

#!/bin/bash

# Default values
ORG="myorg"     # Your organization
PER_PAGE=1000   # per_page maxes out at 1000
TOKEN= "**********"
USE_FLAGS=false # Whether options are provided as flags

# Display usage information
show_help() {
	echo "Usage: "**********"
	echo "Options:"
	echo "  -o, --org      GitHub organization name (default: 'myorg')"
	echo "  -t, --token    GitHub API token (optional)"
	echo "  -h, --help     Show this help message"
	exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
	-o | --org)
		ORG="$2"
		shift 2
		USE_FLAGS=true
		;;
	-t | --token)
		TOKEN= "**********"
		shift 2
		USE_FLAGS=true
		;;
	-h | --help) show_help ;;
	*) break ;;
	esac
done

# If no flags are used, treat the remaining arguments as positional
if [ "$USE_FLAGS" = false ]; then
	if [[ "$1" == "-h" || "$1" == "--help" ]]; then
		show_help
	fi
	ORG=$1
	TOKEN= "**********"
fi

# Main logic
if [[ -n "$TOKEN" ]]; then
	gh auth login --with-token <<<"$TOKEN" >/dev/null
fi

gh repo list "$ORG" --limit "$PER_PAGE" --json visibility,sshUrl,nameWithOwner |
	jq -r '.[]|select(.visibility == "PUBLIC") | .nameWithOwner + " " + .sshUrl' |
	while IFS= read -r repo_info; do
		repo_name=$(echo "$repo_info" | cut -d ' ' -f 1)
		repo_url=$(echo "$repo_info" | cut -d ' ' -f 2)
		echo "Cloning $repo_name from $repo_url"
		git clone "$repo_url" --quiet
	donet clone "$repo_url" --quiet
	done