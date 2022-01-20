#date: 2022-01-20T16:57:55Z
#url: https://api.github.com/gists/cdd65adcf0e028a3c92b4cdb7119cbb1
#owner: https://api.github.com/users/kis9a

function read_file_line_up_example() {
  local file line
  declare -a lines
  file="$1"
  line="$2"
  i=0
  while read -r l; do
    lines+=("$l")
    ((i++))
  done <"$file"

  for ((j = line; 0 < j; j--)); do
    line_type="$(get_line_type_example "${lines[$j - 1]}")"
    if [[ "$line_type" != "not-specified" ]]; then
      arg="$(echo "${lines[$j - 1]}" | cut -f 2 -d " " | sed -e "s/\"//g")"
      if [[ "$line_type" == "data" ]]; then
        arg="data.$arg"
      fi
      cat <<<"$arg"
      break
    fi
  done
}

function get_line_type_example() {
  if [[ "$1" =~ ^match_pattern.* ]]; then
    echo "resource"
  elif [[ "$1" =~ ^2_match_pattern.* ]]; then
    echo "data"
  else
    echo "not-specified"
  fi
}