#date: 2021-12-03T17:07:23Z
#url: https://api.github.com/gists/2ad639e72e220be4fa1b38677f7ad21d
#owner: https://api.github.com/users/johnmurch

cat sample.json| jq -r '(map(keys) | add | unique) as $cols | map(. as $row | $cols | map($row[.])) as $rows | $cols, $rows[] | @csv'  > sample.csv 