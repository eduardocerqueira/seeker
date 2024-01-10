#date: 2024-01-10T16:49:15Z
#url: https://api.github.com/gists/085030d5cf315c83ae2837b2e8ab2580
#owner: https://api.github.com/users/LesterLian

curl -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/markdown \
  -d '{"text":'"$(cat README.md | jq -s -R '.')"'}' 