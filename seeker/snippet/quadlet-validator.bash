#date: 2026-01-23T17:02:14Z
#url: https://api.github.com/gists/daca6599f729e7dbda025f56047f448c
#owner: https://api.github.com/users/mnl

#!/usr/bin/env bash
# vim: set ts=2 sw=2 ft=bash noet :
# shellcheck disable=SC2064 # We want immediate expansion
# Wrapper to validate a single quadlet unit file

set -Eeu
QUADLET_FILE=${1:?Argument missing: Filename}
test -r "$QUADLET_FILE"
QUADLET_BIN=/usr/libexec/podman/quadlet
QUADLET_TMP=$(mktemp -dt quadlet-XXXX)
trap "rm -rf $QUADLET_TMP" EXIT ERR
cp "${QUADLET_FILE}" "${QUADLET_TMP}"
QUADLET_UNIT_DIRS="$QUADLET_TMP" "$QUADLET_BIN" -dryrun >/dev/null && \
    printf "%s[%d]: %s\n" "$(basename "$0")" $$ "No errors found"