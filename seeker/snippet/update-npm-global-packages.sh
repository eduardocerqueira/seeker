#date: 2021-12-23T16:46:33Z
#url: https://api.github.com/gists/22ef3ba55df73f56d71497f9f9810388
#owner: https://api.github.com/users/floydpink

npm ls -g -j --depth=0 | jq -r '.dependencies | keys | join(" ")' | xargs npm install -g