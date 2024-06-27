#date: 2024-06-27T16:57:04Z
#url: https://api.github.com/gists/641d45d3aede2a280fb8c31025fd090b
#owner: https://api.github.com/users/dkirkby

find . -name "*.ipynb" -not -path "*checkpoint*" -print0 | xargs -0 fgrep -l coordinates