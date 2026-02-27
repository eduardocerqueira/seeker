#date: 2026-02-27T17:24:13Z
#url: https://api.github.com/gists/5ae43fa474993b3dae821aa28ca99205
#owner: https://api.github.com/users/jonebarker

for f in *; do
    [ -f "$f" ] || continue
    name="${f%.*}"
    mkdir -p "$name"
    mv "$f" "$name/"
done
