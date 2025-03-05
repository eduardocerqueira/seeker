#date: 2025-03-05T16:48:48Z
#url: https://api.github.com/gists/75628fd9b042f30cfaaf2507338f9b50
#owner: https://api.github.com/users/ccamel

( export CONTRACT="axone1xa8wemfrzq03tkwqxnv9lun7rceec7wuhh8x3qjgxkaaj5fl50zsmj8u0n"; \
  export NODE="https://axone-rpc.jayjayinfra.space/"; \
  axoned \
  --node "$NODE" \
  query wasm contract-state smart "$CONTRACT" \
    '{"store":{}}' \
    -o json \
| jq -r '
  .data.stat as $stat |
  "Triples=" +
    ($stat.triple_count
      | tonumber
      | tostring
      | gsub("(?<=\\d)(?=(\\d{3})+$)"; " ")) +
  "\nSize=" +
    ($stat.byte_size
      | tonumber
      | tostring
      | gsub("(?<=\\d)(?=(\\d{3})+$)"; " "))
' \
| bat -p --language=DotENV )