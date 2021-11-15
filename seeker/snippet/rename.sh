#date: 2021-11-15T16:57:04Z
#url: https://api.github.com/gists/21a2719b9ab1be258f2e62dd361823d8
#owner: https://api.github.com/users/kitzberger

# Recursively renaming all `.ts` files to `.typoscript`
# ; matching subdirectories
# * matching the filename

mmv ";*.ts" "#1#2.typoscript"