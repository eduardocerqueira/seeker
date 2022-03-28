#date: 2022-03-28T17:07:03Z
#url: https://api.github.com/gists/d36037745b7afcdc13c51d079273ac26
#owner: https://api.github.com/users/vsuharnikov

# AS IS
# Based on https://gist.github.com/Nervengift/0ab9e6127ac17b8317ac
# Works with the stacked layout too
ws=$(swaymsg -t get_workspaces|jq "map(select(.focused))[]|.name")
windows=$(swaymsg -t get_tree|jq ".nodes|map(.nodes[])|map(select(.type==\"workspace\" and .name==$ws))[0]|[recurse(.nodes[])|select(.layout==\"none\")]|map({pid: .pid, focused: .focused})")

# https://stackoverflow.com/questions/53135035/jq-returning-null-as-string-if-the-json-is-empty
current=$(echo $windows | jq ".[]|select(.focused).pid // empty")

if [[ -n $current ]]; then
  first=$(echo $windows | jq ".[0]|select(.focused==false).pid")
  after=$(echo $windows | jq 'map(.pid, if .focused then "F" else "" end) | join(" ") | split(" F") | last | split(" ") | map(select(. != "")) | first // empty')

  if [[ -n $after ]]; then
    next=$after
  else
    next=$first
  fi
  swaymsg "[pid=$next]" focus > /dev/null
fi
