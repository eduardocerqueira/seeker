#date: 2025-06-10T16:55:51Z
#url: https://api.github.com/gists/b55c822de66b44b4095d4bb737676ded
#owner: https://api.github.com/users/JonLz

#!/bin/bash
# Usage: ./spm_buster.sh <package_name1> [<package_name2> ...]
#
# This script searches for and removes cache and metadata related to specific Swift packages
# from Xcode DerivedData, SwiftPM cache, and SwiftPM fingerprints.
#
# Arguments:
#   <package_name1> [<package_name2> ...]   One or more package names (case-insensitive).
#
# Example:
#   ./spm_buster.sh pkg_name pkg_name2
#
# The script will display all matches and prompt for confirmation before deleting any files.
set -e

# Directories to search
DERIVED_DATA=~/Library/Developer/Xcode/DerivedData
SWIFTPM_CACHE=~/Library/Caches/org.swift.swiftpm/repositories
SWIFTPM_FINGERPRINTS=~/Library/org.swift.swiftpm/security/fingerprints

# Arrays to hold found items
declare -a derived_matches
declare -a cache_matches
declare -a fingerprint_matches

# Search for each package name
for pkg in "$@"; do
    # DerivedData: match directories containing the package name
    while IFS= read -r -d '' dir; do
        derived_matches+=("$dir")
    done < <(find "$DERIVED_DATA" -type d -iname "*$pkg*" -prune -print0 2>/dev/null)

    # SwiftPM cache: match directories containing the package name
    while IFS= read -r -d '' dir; do
        cache_matches+=("$dir")
    done < <(find "$SWIFTPM_CACHE" -type d -iname "*$pkg*" -prune -print0 2>/dev/null)

    # Fingerprints: match json files containing the package name
    while IFS= read -r -d '' file; do
        fingerprint_matches+=("$file")
    done < <(find "$SWIFTPM_FINGERPRINTS" -type f -iname "*$pkg*.json" -print0 2>/dev/null)
done

# Tabulate results
echo "=== Summary of matches ==="
echo
echo "DerivedData matches:"
for f in "${derived_matches[@]}"; do
    echo -e "\t${f#$DERIVED_DATA/}"
done
echo
echo "SwiftPM cache matches:"
for f in "${cache_matches[@]}"; do
    echo -e "\t${f#$SWIFTPM_CACHE/}"
done
echo
echo "SwiftPM fingerprint matches:"
for f in "${fingerprint_matches[@]}"; do
    echo -e "\t${f#$SWIFTPM_FINGERPRINTS/}"
done
echo

# Confirm deletion
read -p "Do you want to delete all of the above files/directories? (y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    for f in "${derived_matches[@]}"; do
        echo "Deleting $f"
        rm -rf "$f"
    done
    for f in "${cache_matches[@]}"; do
        echo "Deleting $f"
        rm -rf "$f"
    done
    for f in "${fingerprint_matches[@]}"; do
        echo "Deleting $f"
        rm -f "$f"
    done
    echo "Deletion complete."
else
    echo "No files were deleted."
fi