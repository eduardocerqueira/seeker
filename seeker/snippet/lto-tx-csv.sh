#date: 2022-10-11T17:15:49Z
#url: https://api.github.com/gists/1bdacc967d90f37b75a275b340a1f0d6
#owner: https://api.github.com/users/jasny

ADDRESS="$1"

curl -s "https://node1.lto.network/index/transactions/addresses/$ADDRESS" -o "$ADDRESS.json"

jq -r 'reverse | .[] | [.type, .version, .id, (.timestamp / 1000 | todateiso8601), .sender, .recipient, (.amount // 0) + ((.transfers // []) | ([.[].amount] | add)) // 0, .effectiveFee] | join(",")' < "$ADDRESS.json" > "$ADDRESS.csv"