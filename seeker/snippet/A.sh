#date: 2025-05-20T17:11:09Z
#url: https://api.github.com/gists/bcaf667e6a66621cd2805887d883be26
#owner: https://api.github.com/users/casper1zxy

version="2.55.1"  # your target version

last_modified=$(
  curl -s -u "$user: "**********"
    .files[]
    | select(.uri == ("/" + $version + "/manifest.json"))
    | .lastModified
  '
)

echo "Last modified of manifest.json for version $version: $last_modified"on for version $version: $last_modified"