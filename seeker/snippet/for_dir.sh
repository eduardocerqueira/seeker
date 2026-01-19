#date: 2026-01-19T17:18:41Z
#url: https://api.github.com/gists/d7e7262bba8be8106f2138706be13d5b
#owner: https://api.github.com/users/tiation

# Install: add this function to your ~/.bashrc or ~/.profile
# Open a new shell and use it:
#
# $ for_dir /tmp /home /var -- ls -la
#

function for_dir()
{
  local dirs=()

  if ! [[ "$*" =~  ' -- ' ]]; then
    echo "Usage: $0 DIRS... -- COMMAND"
    return 1
  fi  

  while [ $# -gt 0 ]; do
    case "$1" in
      --) 
        shift
        break
      ;;  
      *)  
        dirs+=( "$1" )
    esac
    shift
  done

  for dir in "${dirs[@]}"; do
    cd "$dir"
    "$@"
    cd -
  done
}