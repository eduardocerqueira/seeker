#date: 2025-12-01T17:06:09Z
#url: https://api.github.com/gists/3e232e0123ea5220c479209a6d4964c0
#owner: https://api.github.com/users/killeik

cut-with-ansi(){
  visible_limit="$1"
  line="$2"
  output=""
  visible_count=0
  i=0; len=${#line}
  
  while (( i < len && visible_count < visible_limit )); do
      char="${line:i:1}" 
      if [[ $char == $'\e' ]]; then
          output+="$char"; ((i++))
          # copy until 'm'
          while (( i < len )); do
              char="${line:i:1}"
              output+="$char"
              ((i++))
              [[ $char == "m" ]] && break
          done
      else
          output+="$char"
          ((visible_count++))
          ((i++))
      fi
  done
  printf '%s\n' "$output"
}