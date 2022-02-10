#date: 2022-02-10T16:43:00Z
#url: https://api.github.com/gists/cec1d4ecfb729ad95c788d53df8dd7f1
#owner: https://api.github.com/users/oirodolfo

# Reusable bash function you can add to your ~/.zshrc or ~/.bashrc file
#
# Usage: pkg-script start "node index.js"
#
function pkg-script () {
  echo $(jq --arg key "${1}" --arg val "${2}" '.scripts[$key]=$val' package.json) | jq . | > package.json
}