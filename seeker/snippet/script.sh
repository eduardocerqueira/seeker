#date: 2025-11-24T16:56:27Z
#url: https://api.github.com/gists/a54dbcdf7117e3f1c5e861fadbac0405
#owner: https://api.github.com/users/angeloevangelista

#!/usr/bin/env zsh

for FOLDER in *(/); do
  if [[ -d "$FOLDER/.git" ]]; then
    cd "$FOLDER" || continue

    MODIFIED_FILES=("${(@f)$(git ls-files -m)}")

    if (( ${#MODIFIED_FILES[@]} > 0 )); then
      echo "Applying chmod 644 in $FOLDER"
      chmod 644 -- "${MODIFIED_FILES[@]}"
    fi

    cd - >/dev/null || exit
  fi
done
