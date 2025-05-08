#date: 2025-05-08T16:59:19Z
#url: https://api.github.com/gists/862c59678d3aef3eaa4b8ffb7589aac2
#owner: https://api.github.com/users/lmmx

cargo tree -e no-dev --duplicates --depth 0 | sed '/^$/d' | grep -v 'dependencies]' | uniq | cut -d ' ' -f 1 | uniq -d