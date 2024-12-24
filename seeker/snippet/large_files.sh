#date: 2024-12-24T17:02:01Z
#url: https://api.github.com/gists/0d3c21f0cb0c9a432536be9c39e39008
#owner: https://api.github.com/users/jkeefe

# example: get rid of large `.tif` and `.nc` files in gis/tmp
# from https://www.geeksforgeeks.org/how-to-remove-a-large-file-from-commit-history-in-git/

git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch gis/tmp/*.tif' \
--prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch gis/tmp/*.nc' \
--prune-empty --tag-name-filter cat -- --all

git push origin --force --all

# Caveat: Force pushing rewrites history on the remote server, which can disrupt other
# developers who have already pulled that branch.