#date: 2025-12-17T17:14:46Z
#url: https://api.github.com/gists/48fab47f491b1f3eec0bba71370f1ee6
#owner: https://api.github.com/users/nicobailon

# ZSH version - add to ~/.zshrc

p() {
  local cache=~/.cache/pi-update-available
  [[ -f $cache ]] && { npm i -g @mariozechner/pi-coding-agent; rm $cache; }
  { 
    local local=$(jq -r .version /opt/homebrew/lib/node_modules/@mariozechner/pi-coding-agent/package.json 2>/dev/null)
    local remote=$(npm view @mariozechner/pi-coding-agent version 2>/dev/null)
    [[ "$remote" && "$remote" != "$local" ]] && touch $cache
  } &>/dev/null &!
  pi "$@"
}
