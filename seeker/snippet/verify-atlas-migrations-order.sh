#date: 2025-06-04T17:09:10Z
#url: https://api.github.com/gists/ebec004cab078ef12a0f126b2660ce90
#owner: https://api.github.com/users/joeycumines

#!/bin/sh

# verify-atlas-migrations-order - Guards a subset of rewritten migration issues.
#
# USAGE:
#   verify-atlas-migrations-order base_ref head_ref [migration_dir]
#
# If you don't specify a migration directory, it defaults to `migrations`.

if ! {
  [ "$#" -ge 2 ] &&
    base_ref="$1" &&
    [ -n "$base_ref" ] &&
    head_ref="$2" &&
    [ -n "$head_ref" ] &&
    migration_dir="${3:-migrations}" &&
    sum_file="$migration_dir/atlas.sum" &&
    base_sum="$(git show "$base_ref":"$sum_file")" &&
    [ -n "$base_sum" ] &&
    head_sum="$(git show "$head_ref":"$sum_file")" &&
    [ -n "$head_sum" ]
}; then
  echo "Usage: $0 base_ref head_ref [migration_dir]" >&2
  exit 2
fi

# diff across just the *.sql files in the migration directory
if ! {
  migration_diff="$(git diff --name-only "$base_ref..$head_ref" -- "$migration_dir")" &&
    migration_diff="$(printf '%s\n' "$migration_diff" | sed -n '/\.sql$/p')"
}; then
  echo "Failed to get migration diff from base ref '$base_ref' to head ref '$head_ref' in directory '$migration_dir'" >&2
  exit 1
fi

if [ -z "$migration_diff" ]; then
  # no *.sql changes, success, don't print anything (reduces irrelevant noise)
  exit 0
fi

# fail if there are any (unmerged) base_ref changes
if ! unmerged_log="$(git log --oneline "$head_ref..$base_ref" -- "$migration_dir")"; then
  echo "Failed to get unmerged log from base ref '$base_ref' to head ref '$head_ref' in directory '$migration_dir'" >&2
  exit 1
fi
if ! [ -z "$unmerged_log" ]; then
  printf 'There are unmerged changes in the base ref '\''%s'\'' not in head ref '\''%s'\'' for the migration directory '\''%s'\'':\n%s' "$base_ref" "$head_ref" "$migration_dir" "$unmerged_log" >&2
  exit 1
fi

# Compare the two atlas.sum files, line by line.

# set IFS to newline (we will use word expansion to split lines)
IFS='
'

echo_pop_last() {
  if [ "$#" -gt 0 ]; then
    printf '%s' "$1"
    shift
    while [ "$#" -gt 1 ]; do
      printf ' %s' "$1"
      shift
    done
    echo
  fi
}

check_atlas_sum_files() {
  if ! { [ "$#" -ge 1 ] && [ -n "$1" ]; }; then
    echo "Unexpected atlas.sum file check: invalid '$sum_file' for '$base_ref'" >&2
    return 1
  fi

  any=0
  changed=0

  for line in ${head_sum}; do
    if ! { [ "$#" -eq 0 ] || [ -n "$1" ]; }; then
      echo "Unexpected empty line in '$sum_file' for '$base_ref'" >&2
      return 1
    fi

    if ! [ -n "$line" ]; then
      echo "Unexpected empty line in '$sum_file' for '$head_ref'" >&2
      return 1
    fi

    # the first line needs to _differ_ between the two refs, since we had sql changes
    if [ "$any" -eq 0 ]; then
      if ! {
        [ "$#" -gt 0 ] &&
          [ "$1" != "$line" ]
      }; then
        echo "The first line of '$sum_file' for '$base_ref' does not differ from that of '$head_ref'" >&2
        return 1
      fi
      any=1
      shift
      continue
    fi

    # the first line changed confirms a different migration
    # (N.B. might be the same one rewritten, but it is definitely a change)
    if [ "$changed" -eq 0 ]; then
      if [ "$#" -eq 0 ]; then
        changed=1
      elif [ "$1" = "$line" ]; then
        shift
      else
        changed=1
      fi
      continue
    fi

    # Since we've encountered a changed line (other than the header) we
    # expect all subsequent lines to represent files that are _changed_.
    # The presence of any unchanged migration files indicates that
    # an out-of-order migration may have been introduced.
    if ! {
      IFS=' ' migration_file="$(echo_pop_last ${line})" &&
        [ -n "$migration_file" ] &&
        [ "$line" != "$migration_file" ] &&
        printf '%s\n' "$migration_diff" | grep -qxF "$migration_dir/$migration_file"
    }; then
      echo "Out-of-order migration file in '$sum_file' for '$head_ref': $line" >&2
      return 1
    fi
  done

  if [ "$any" -eq 0 ]; then
    echo "Unexpected content in '$sum_file' from '$base_ref' to '$head_ref'" >&2
    return 1
  fi

  if [ "$changed" -eq 0 ]; then
    echo "No changed migration files in '$sum_file' from '$base_ref' to '$head_ref'" >&2
    return 1
  fi

  # just validate that the remaining base_ref lines are sane... not super useful but mah completeness...
  for line in "$@"; do
    if ! [ -n "$line" ]; then
      echo "Unexpected empty line in '$sum_file' for '$base_ref'" >&2
      return 1
    fi
  done
}

# handles the final exit code
check_atlas_sum_files ${base_sum}
