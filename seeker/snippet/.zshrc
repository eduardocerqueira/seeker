#date: 2023-04-17T16:45:05Z
#url: https://api.github.com/gists/70734325a0c1bdc373b187839096c3f6
#owner: https://api.github.com/users/jmkellenberger

function note {
  YEAR=$(date +"%Y")
  DATE=$(date +"%Y-%m-%d")
  TS=$(date +"%Y-%m-%d %H:%M:%S")

  if [ -d .notes ]; then
    FILE=".notes/$YEAR/$DATE.md"
  else
    FILE="$HOME/.notes/$YEAR/$DATE.md"
  fi

  if [ ! -d "$(dirname "$FILE")" ]; then
    mkdir -p "$(dirname "$FILE")"
  fi

  TITLE="# $(date +"%Y-%m-%d %H:%M:%S") - $@"
  
  echo "$TITLE" >> "$FILE"
  
  hx "$FILE"
}

function notes {
  YEAR=$(date +"%Y")
  DATE=$(date +"%Y-%m-%d")

  if [ -d ./.notes ]; then
    FILE="./.notes/$YEAR/$DATE.md"
  else
    FILE="$HOME/.notes/$YEAR/$DATE.md"
  fi

  if [ -f "$FILE" ]; then
    cat "$FILE"
  else
    echo "No notes for today"
  fi
}

function all-notes {
  if [ -d ./.notes ]; then
    NOTES_DIR="./.notes"
  else
    NOTES_DIR="$HOME/.notes"
  fi

  find "$NOTES_DIR" -type f -name "*.md" -exec cat {} \; | less
}

function note-search {
  all-notes | rg "$@"
}
