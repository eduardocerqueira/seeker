#date: 2021-10-20T17:03:34Z
#url: https://api.github.com/gists/bd6e3c4c75c85ad4c3a8e6fea640792e
#owner: https://api.github.com/users/wknapik

# Requires: bash, coreutils, git.

# Print link(s) to file(s) in a repository, with optional line numbers.
# $@ := file_path[:file_line] [ file_path2[:file_line2] [ ... ] ]
repolink() {
    local branch f file host origin
    for f in "$@"; do
        file="$(realpath "$f")"
        checkout_path="$(git -C "$(dirname "$file")" rev-parse --show-toplevel)"
        origin="$(git -C "$checkout_path" config --get remote.origin.url)"
        branch="$(git -C "$checkout_path" rev-parse --abbrev-ref HEAD)"
        host="${origin#*://}"       # strip off scheme
        host="${host#*@}"           # strip off user
        host="${host/:[0-9]*\//\/}" # strip off port
        host="${host//://}"         # replace : with /
        host="${host%.git}"         # strip off .git suffix
        host="${host%/}"            # strip off trailing /

        url_path="${file#"$checkout_path"/}"
        echo "https://$host/blob/$branch/${url_path/:/#L}"
    done
}