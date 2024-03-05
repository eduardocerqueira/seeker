#date: 2024-03-05T17:10:34Z
#url: https://api.github.com/gists/3f6f122d84a974735abd0be4186bfa7d
#owner: https://api.github.com/users/AdjectiveAllison

#!/usr/bin/env bash

function display_help {
  echo "Usage: $0 [options] [directory]"
  echo
  echo "Options:"
  echo "  -h, --help           Display this help message"
  echo "  -e, --extension      Filter files by extension (comma-separated)"
  echo "  -n, --name           Filter files by name (regex pattern)"
  echo "  -t, --no-tree        Do not print the tree list of included files"
  echo "  !<directory>         Ignore specified directory"
  echo
  echo "Examples:"
  echo "  $0                                  Process files in the current directory"
  echo "  $0 /path/to/directory               Process files in the specified directory"
  echo "  $0 -e txt,md                        Process only .txt and .md files"
  echo "  $0 -n '^prefix'                     Process only files with names starting with 'prefix'"
  echo "  $0 -e txt -n '^prefix' !ignore_dir  Process .txt files with names starting with 'prefix', ignoring 'ignore_dir'"
  exit 0
}

function should_ignore {
  local entry="$1"
  local gitignore_path="$2"
  local ignore_list=("${@:3}")

  if [[ -f "$gitignore_path" ]]; then
    while IFS= read -r pattern; do
      if [[ "$pattern" == "" || "$pattern" == "#"* ]]; then
        continue
      fi
      if [[ "$entry" == $pattern* ]]; then
        return 0
      fi
    done < "$gitignore_path"
  fi

  for pattern in "${ignore_list[@]}"; do
    if [[ "$entry" == $pattern* ]]; then
      return 0
    fi
  done

  return 1
}

function should_process {
  local file="$1"
  local extension_filter="$2"
  local name_filter="$3"

  if [[ -n "$extension_filter" ]]; then
    local file_extension="${file##*.}"
    local extensions=(${extension_filter//,/ })
    local match=0
    for ext in "${extensions[@]}"; do
      if [[ "$file_extension" == "$ext" ]]; then
        match=1
        break
      fi
    done
    if [[ $match -eq 0 ]]; then
      return 1
    fi
  fi

  if [[ -n "$name_filter" ]]; then
    if [[ ! "$file" =~ $name_filter ]]; then
      return 1
    fi
  fi

  return 0
}

function process_file {
  local file="$1"
  local relative_path="${file#$target_directory/}"
  local extension_filter="$2"
  local name_filter="$3"

  if [[ -r "$file" ]] && file -b --mime-type "$file" | grep -q '^text/' && should_process "$file" "$extension_filter" "$name_filter"; then
    echo "<file>"
    echo "<path>$relative_path</path>"
    echo "<contents>"
    if cat "$file"; then
      echo "</contents>"
      echo "</file>"
      included_files["$relative_path"]=1
      local dir=$(dirname "$relative_path")
      while [[ "$dir" != "." ]]; do
        directory_order["$dir"]=1
        dir=$(dirname "$dir")
      done
    else
      echo "Error reading file: $file"
      echo "</file>"
    fi
  fi
}

function process_directory {
  local directory="$1"
  local gitignore_path="$directory/.gitignore"
  local extension_filter="$2"
  local name_filter="$3"
  local ignore_list=("${@:4}")

  for entry in "$directory"/*; do
    local relative_entry="${entry#$directory/}"
    if [[ -f "$entry" ]] && ! should_ignore "$relative_entry" "$gitignore_path" "${ignore_list[@]}"; then
      process_file "$entry" "$extension_filter" "$name_filter"
    elif [[ -d "$entry" ]] && ! should_ignore "$relative_entry/" "$gitignore_path" "${ignore_list[@]}"; then
      process_directory "$entry" "$extension_filter" "$name_filter" "${ignore_list[@]}"
    fi
  done
}

function print_tree_list {
  local directory="$1"
  local indent="$2"
  local current_dir=""

  for dir in "${!directory_order[@]}"; do
    if [[ "$dir" != "." ]]; then
      echo "${indent}${dir#$directory/}/"
    fi
    for file in "${!included_files[@]}"; do
      if [[ "$(dirname "$file")" == "$dir" ]]; then
        echo "${indent}  $(basename "$file")"
      fi
    done
  done

  for file in "${!included_files[@]}"; do
    if [[ "$(dirname "$file")" == "." ]]; then
      echo "${indent}$(basename "$file")"
    fi
  done
}

target_directory="."
extension_filter=""
name_filter=""
ignore_list=()
print_tree=true
declare -A included_files
declare -A directory_order

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      display_help
      ;;
    -e|--extension)
      shift
      extension_filter="$1"
      ;;
    -n|--name)
      shift
      name_filter="$1"
      ;;
    -t|--no-tree)
      print_tree=false
      ;;
    !*)
      ignore_list+=("${1:1}")
      ;;
    *)
      if [[ -z "$target_directory" ]] && [[ -d "$1" ]]; then
        target_directory="$1"
      else
        echo "Unknown option: $1"
        echo "Use -h or --help for usage information"
        exit 1
      fi
      ;;
  esac
  shift
done

process_directory "$target_directory" "$extension_filter" "$name_filter" "${ignore_list[@]}"

if $print_tree; then
  echo
  echo "Tree list of included files:"
  print_tree_list "$target_directory" ""
fi