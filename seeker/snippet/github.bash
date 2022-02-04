#date: 2022-02-04T16:52:36Z
#url: https://api.github.com/gists/9f182512bab731f3b3b540b9bba4fdb1
#owner: https://api.github.com/users/johnloy

alias gh="open \`git remote -v | grep git@github.com | grep fetch | head -1 | cut -f2 | cut -d' ' -f1 | sed -e's/:/\//' -e 's/git@/http:\/\//'\`"