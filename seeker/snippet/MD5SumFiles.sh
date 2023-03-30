#date: 2023-03-30T16:52:09Z
#url: https://api.github.com/gists/7c3e8452d9fbe888f8fb5e68db178d8c
#owner: https://api.github.com/users/DeltaEchoSierra

find "$PWD" -type d | sort | while read dir; do cd "${dir}"; [ ! -f @md5Sum.md5 ] && echo "Processing " "${dir}" || echo "Skipped " "${dir}" " @md5Sum.md5 allready present" ; [ ! -f @md5Sum.md5 ] &&  md5sum * > @md5Sum.md5 ; chmod a=r "${dir}"/@md5Sum.md5 ;done