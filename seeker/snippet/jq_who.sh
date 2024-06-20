#date: 2024-06-20T17:05:48Z
#url: https://api.github.com/gists/81b9d034d0752f680ae381e52028e681
#owner: https://api.github.com/users/derekhecksher

curl https://ipinfo.io | duckdb -c "select * from read_json_auto('/dev/stdin')"