#date: 2024-08-29T16:55:08Z
#url: https://api.github.com/gists/11e26db32d5c7a870476954a1cd40666
#owner: https://api.github.com/users/pa-0

zip -rX "../$(basename "$(realpath .)").epub" mimetype $(ls|xargs echo|sed 's/mimetype//g')