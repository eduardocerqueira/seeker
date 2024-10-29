#date: 2024-10-29T17:01:49Z
#url: https://api.github.com/gists/c1149db9172a0dfb000e532d5aa90405
#owner: https://api.github.com/users/fbicknel

# Simply copy the file with the date stuck on the end
backup_file () {
  cp -vp "${1}" "${1}.$(date '+%y%m%d%H%M')"
}