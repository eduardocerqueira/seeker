#date: 2022-04-20T17:08:16Z
#url: https://api.github.com/gists/d140107518c95eb58018a899f9188003
#owner: https://api.github.com/users/dcxSt

for file in *; do
  if [[ ${file: -5} == ".json" ]]; then
    jq . $file > temp.json
    cp temp.json $file
    rm temp.json
    echo "$file is now pretty"
  else
    echo "\t$file not valid json"
  fi
done
