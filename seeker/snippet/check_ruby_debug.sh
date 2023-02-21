#date: 2023-02-21T17:06:13Z
#url: https://api.github.com/gists/8e3690512b7cf95d487d5425acb083f4
#owner: https://api.github.com/users/wlads

# gem list -i -e 'debug' -v '>= 1.0.0'
# -i returns true / false
# -e exact match (avoid partial match), could also be a regex e.g. '^debug$'
# -v specify gem version (can use pessimistic version)

function check_ruby_debug() {
  if [[ ! $(gem list -i -e "debug" -v ">= 1.0.0") ]]; then
    echo -e "Debug gem not installed!"
    echo -e "run: gem install debug"
    exit 1
  fi
}