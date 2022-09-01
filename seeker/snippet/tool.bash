#date: 2022-09-01T17:01:19Z
#url: https://api.github.com/gists/96abfd71bb8d330a495fc695b764dfe6
#owner: https://api.github.com/users/ciis0

#!/usr/bin/env bash

SCRIPT_CALL=${BASH_SOURCE[0]}
SCRIPT_REAL=$(realpath $SCRIPT_CALL)

name=$(basename $SCRIPT_REAL)
ver_name=${name^^}_VER
ver_default_name=${ver_name}_DEFAULT

ver=${!ver_name:-${!ver_default_name}}

opt=$(dirname $SCRIPT_REAL)/../opt
bin=bin/$(basename $SCRIPT_CALL)

if [ -d $opt/$name-$ver ]
then $opt/$name-$ver/$bin "$@"
else $opt/$name-${ver}*/$bin "$@"
fi
