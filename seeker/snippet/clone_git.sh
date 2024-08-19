#date: 2024-08-19T16:52:13Z
#url: https://api.github.com/gists/d048ff367e8ca0e05f93cb55f8785b98
#owner: https://api.github.com/users/damaon

gh_print() {
    local github_url="$1"
    local repo_name=$(basename "$github_url" .git)
    local tmp_dir="/tmp/$repo_name"

    # Clone the repository
    if git clone "$github_url" "$tmp_dir" > /dev/null 2>&1; then
        : # Silent success
    else
        echo "Failed to clone repository"
        return 1
    fi

    # Define the tree_with_content function
    tree_with_content() {
        local dir="${1:-.}"  # Use current directory if no argument is provided
        local base_dir="$(cd "$dir" && pwd)"
        find "$dir" -type f -not -path '*/\.git/*' | while read -r file; do
            local relative_path="${file#$base_dir/}"
            echo "[$relative_path]"
            # Check if the file is a text file
            if file -b --mime-type "$file" | grep -q "^text/"; then
                cat "$file"
            else
                echo "(Not a text file)"
            fi
            echo ""
        done
    }

    # Call tree_with_content on the cloned repository
    tree_with_content "$tmp_dir"

    # Optional: Remove the cloned repository to clean up
    # Uncomment the next line if you want to delete the cloned repo after printing
    # rm -rf "$tmp_dir"
}
