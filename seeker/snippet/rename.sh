#date: 2023-04-03T16:53:54Z
#url: https://api.github.com/gists/96ff5264fb91a62e1c024133d0c8277a
#owner: https://api.github.com/users/jameshfisher

rename() {
  [ "$#" -ne 1 ] && echo "Usage: rename <file_path>" && return 1
  local temp_file=$(mktemp)
  echo "$1" > "$temp_file"
  "${EDITOR:-vi}" "$temp_file"
  local new_path=$(cat "$temp_file")
  mv "$1" "$new_path"
  echo "$new_path"
  rm "$temp_file"
}