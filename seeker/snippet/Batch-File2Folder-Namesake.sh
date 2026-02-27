#date: 2026-02-27T17:20:40Z
#url: https://api.github.com/gists/e2267c99dee70a0e0a3fa8a6d4ff7ad3
#owner: https://api.github.com/users/jonebarker

for f in *; do
    [ -f "$f" ] || continue
    name="${f%.*}"
    mkdir -p "$name"
done
