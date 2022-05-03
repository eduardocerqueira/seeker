#date: 2022-05-03T17:21:55Z
#url: https://api.github.com/gists/8a0bbec9a95507c19be944efb80f70f7
#owner: https://api.github.com/users/goququ

# Format staged js/ts files before commit
npx prettier --write $(git ls-files --modified -u | awk '{ print $4 }' | grep -E -i '.(t|j)sx?')