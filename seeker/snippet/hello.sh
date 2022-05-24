#date: 2022-05-24T17:07:53Z
#url: https://api.github.com/gists/ab998aad575f0636e339dd564049446b
#owner: https://api.github.com/users/dotysan

#! /usr/bin/env bash
#
# A template for seeding new Bash scripts.
#
if [ "${BASH_VERSINFO[0]}" != "5" ]
then
    echo "ERROR: Only tested on Bourne-Again SHell v5."
    exit 1
fi >&2
set -ex
shopt -s nullglob
now=$(date -Im)

main() {
    foo
}
########################################################################

foo() {
    echo bar
}

########################################################################
# poor man's __main__
return 2>/dev/null ||:
main "$@"
exit 0
