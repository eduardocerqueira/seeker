#date: 2023-01-26T17:00:46Z
#url: https://api.github.com/gists/2f7f9b76272830d1f2535930b80388f6
#owner: https://api.github.com/users/iuliaL

find . -iname "*.txt" -exec bash -c 'mv "$0" "${0%\.txt}.md"' {} \;
