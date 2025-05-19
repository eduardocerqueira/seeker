#date: 2025-05-19T16:54:54Z
#url: https://api.github.com/gists/229e60cd5d05d657f9f060264942a5c3
#owner: https://api.github.com/users/rmhaiderali

#!/bin/bash

cmd="$1"
input1="$2"
input2="$3"

process() {
  tmp="${cmd//@1/$1}"
  eval "${tmp//@2/$2}"
}

is_file() {
  [[ -f "$1" ]]
}

is_dir() {
  [[ -d "$1" ]]
}

if is_file "$input1" && [[ -z "$input2" ]]; then
  process "$input1"

elif is_dir "$input1" && [[ -z "$input2" ]]; then
  for file in "$input1"/*; do
    [[ -f "$file" ]] && process "$file"
  done

elif is_file "$input1" && is_file "$input2"; then
  process "$input1" "$input2"

elif is_file "$input1" && is_dir "$input2"; then
  for file in "$input2"/*; do
    [[ -f "$file" ]] && process "$input1" "$file"
  done

elif is_dir "$input1" && is_file "$input2"; then
  for file in "$input1"/*; do
    [[ -f "$file" ]] && process "$file" "$input2"
  done

elif is_dir "$input1" && is_dir "$input2"; then
  for file1 in "$input1"/*; do
    [[ -f "$file1" ]] || continue
    for file2 in "$input2"/*; do
      [[ -f "$file2" ]] || continue
      process "$file1" "$file2"
    done
  done

else
  echo "Invalid file or directory"
  exit 1
fi
