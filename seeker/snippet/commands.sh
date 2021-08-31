#date: 2021-08-31T03:07:41Z
#url: https://api.github.com/gists/957bf11da1afbe844d20547186b7fc0e
#owner: https://api.github.com/users/zacwhy

# Remove double quotes at the beginning and end
# Remove escape of double quotes
sed -e '1 s/^.//; $ s/.$//; s/\\\"/"/g' < a.csv