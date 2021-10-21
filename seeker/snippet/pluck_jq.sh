#date: 2021-10-21T17:15:37Z
#url: https://api.github.com/gists/a6aecd22028fc3bbb75a361b58cba7a5
#owner: https://api.github.com/users/monkpit

echo '[
  {"a": "a", "b": "b", "c": "c"},
  {"a": "1", "b": "2", "c": "3"},
  {"a": "x", "b": "y", "c": "z"}
]' | jq ".[] | {b, c}"

# output:
# {
#   "b": "b",
#   "c": "c"
# }
# {
#   "b": "2",
#   "c": "3"
# }
# {
#   "b": "y",
#   "c": "z"
# }