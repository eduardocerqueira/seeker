#date: 2024-09-13T17:09:42Z
#url: https://api.github.com/gists/f44e38e813332db3cd041fe241642ff7
#owner: https://api.github.com/users/creachadair

#!/usr/bin/env bash
#
# Usage: remove-taint.sh <filename>
#
# Remove the annoying mark-of-the-web taint xattrs from a file.
# For some reason macOS ignores a removexattr for these attributes,
# but the taint does not survive a pipe copy and rename.
#
set -euo pipefail

for p in "$@" ; do
    perm="$(stat -f '%Lp' "$p")"    # low-order permission bits, e.g., 644
    t="$(mktemp "$p".XXXXXXXXXXXX)" # in the same directory as the source
    cat "$p" > "$t"

    # Preserve the access bits and modification time from the original.
    chmod "$perm" "$t"
    touch -m -r "$p" "$t"

    # Note: It's not sufficent to just rename, because macOS appears to track
    # the quarantine through the filename even if it's replaced. By removing
    # the original file first, we keep the metadata for the untainted copy.
    # Seriously, Apple, WTAF.
    rm -f -- "$p"
    mv -f -- "$t" "$p"
done
