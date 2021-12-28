#date: 2021-12-28T16:45:22Z
#url: https://api.github.com/gists/94d5efa8f25a388c0d4ebf425a26b6fc
#owner: https://api.github.com/users/luisgagocasas

# watch the current directory for changes to files
# order the files by size and output the files in 3 columns
# highlight changes between outputs

watch -d 'du -sh * | sort -hr | pr -3 -t'