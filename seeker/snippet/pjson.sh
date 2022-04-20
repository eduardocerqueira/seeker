#date: 2022-04-20T17:08:16Z
#url: https://api.github.com/gists/d140107518c95eb58018a899f9188003
#owner: https://api.github.com/users/dcxSt

if [[ ${1: -5} == ".json" ]]; then
  jq . $1 > temp.json       # put nice json into temp file
  cp temp.json $1           # copy it into original file
  rm temp.json              # remove temp file
  echo "Success! $1 is now pretty"
else
  echo "bad arg: '$1', must end in .json"
fi