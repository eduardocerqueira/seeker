#date: 2021-09-30T17:13:11Z
#url: https://api.github.com/gists/a12a4930d9b95778b8854d9d852ff22b
#owner: https://api.github.com/users/libjared

# Usage: yt_ids [DIRECTORY]
# Print out all the youtube video IDs for which yt-dlp content is located immediately within DIRECTORY.
# Arguments
# - DIRECTORY - the directory to search for yt-dlp content. Defaults to ".".
function yt_ids {
  find ${1-.} -maxdepth 1 -type f -printf '%f\n' \
  | grep -Po '(?<= \[)[a-zA-Z0-9_-]{11}(?=\]\.)' \
  | sort -u
}