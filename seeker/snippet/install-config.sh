#date: 2022-01-31T17:07:33Z
#url: https://api.github.com/gists/e014a39d45920ff443a38ec6da32cab0
#owner: https://api.github.com/users/jottr

set -x

cat karabiner.yaml | ruby -r yaml -r json -e 'puts YAML.load($stdin.read).to_json'  \
  | jq --sort-keys 'del(.definitions)' > karabiner.json
