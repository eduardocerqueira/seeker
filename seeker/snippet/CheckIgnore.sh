#date: 2026-02-24T17:40:22Z
#url: https://api.github.com/gists/aaa45c28d0ba38cde82f9ef2fd577ecf
#owner: https://api.github.com/users/maoyeedy

- name: Fail if tracked files are ignored (except debug symbols)
  shell: bash
  run: |
    set -euo pipefail

    # Find tracked files that match ignore rules
    bad_files=$(git ls-files --cached --ignored --exclude-standard 2>/dev/null || true)

    if [ -z "$bad_files" ]; then
      exit 0
    fi

    # Only allow common debug symbol files
    allow_pattern='\.([Pp][Dd][Bb]|[Mm][Dd][Bb]|[Dd][Bb][Gg]|dSYM)(\.meta)?$'

    bad_filtered=$(printf '%s\n' "$bad_files" | grep -v -E "$allow_pattern" || true)

    if [ -z "$bad_filtered" ]; then
      exit 0
    fi

    echo "Error: Tracked files are ignored by .gitignore but not debug symbols:"
    echo "──────────────────────────────────────────────"
    printf '%s\n' "$bad_filtered" | sed 's/^/  - /'
    echo "──────────────────────────────────────────────"
    echo
    echo "Fix:"
    echo "  git rm --cached <file>              # single file"
    echo "  git rm -r --cached dir/             # folder"
    echo "  git commit -m 'Untrack ignored files'"
    echo
    echo "Files remain on disk but are no longer tracked."
    exit 1