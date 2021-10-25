#date: 2021-10-25T17:08:40Z
#url: https://api.github.com/gists/2a4709fea0d3bf6be199bfa3320c7d7c
#owner: https://api.github.com/users/YSaxon

#modified to work inline on stdin
sed "s/[$(echo -ne '\u00B4\u2018\u2019')]/'/g; s/[$(echo -ne '\u201C\u201D')]/\"/g"