#date: 2023-08-21T16:59:36Z
#url: https://api.github.com/gists/75f1a5668d2d6caee415d87f85f82b02
#owner: https://api.github.com/users/C0axx

#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "[*] Creating temporary directory..."
TMPDIR="$(mktemp -d --suffix=_bloodhound-customqueries)"

# Compass BloodHound Customqueries
echo "[*] Downloading Compass BloodHound customqueries..."
curl -s -o "$TMPDIR/customqueries-compass.json" "https://raw.githubusercontent.com/CompassSecurity/BloodHoundQueries/master/BloodHound_Custom_Queries/customqueries.json"

echo "[*] Modifying category on Compass BloodHound customqueries..."
jq '.queries[].category |= (sub("^";"🧭 Compass: "))' < "$TMPDIR/customqueries-compass.json" > "$TMPDIR/customqueries-01-compass-modified.json"

# ZephrFish BloodHound Customqueries
echo "[*] Downloading ZephrFish BloodHound customqueries..."
curl -s -o "$TMPDIR/customqueries-compass.json" "https://raw.githubusercontent.com/ZephrFish/Bloodhound-CustomQueries/main/customqueries.json"

echo "[*] Modifying category on ZephrFish BloodHound customqueries..."
jq '.queries[].category |= (sub("^";"☠️ ZephrFish: "))' < "$TMPDIR/customqueries-compass.json" > "$TMPDIR/customqueries-01-zephrfish-modified.json"

# Certipy BloodHound Customqueries
echo "[*] Downloading Certipy BloodHound customqueries..."
curl -s -o "$TMPDIR/customqueries-certipy.json" "https://raw.githubusercontent.com/ly4k/Certipy/main/customqueries.json"

echo "[*] Modifying category on Certipy BloodHound customqueries..."
jq '.queries[].category |= (sub("^";"🔏 Certipy: "))' < "$TMPDIR/customqueries-certipy.json" > "$TMPDIR/customqueries-02-certipy-modified.json"

# Hausec BloodHound Customqueries
echo "[*] Downloading Hausec BloodHound customqueries..."
curl -s -o "$TMPDIR/customqueries-hausec.json" "https://raw.githubusercontent.com/hausec/Bloodhound-Custom-Queries/master/customqueries.json"

echo "[*] Adding category to Hausec BloodHound customqueries..."
jq '.queries[] |= { "category": "💻 Hausec" } +. ' < "$TMPDIR/customqueries-hausec.json" > "$TMPDIR/customqueries-02-hausec-modified.json"

echo "[*] Merging queries..."
cat "$TMPDIR/"*-modified.json | jq -s 'add + {queries: map(.queries[])}' > customqueries.json

echo "[*] Done. Please copy to your config directory:"
echo "cp customqueries.json ~/.config/bloodhound/"

echo "[*] Bye."
