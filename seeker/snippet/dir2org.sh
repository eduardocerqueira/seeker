#date: 2021-10-07T16:55:22Z
#url: https://api.github.com/gists/b74b6141d3ebe89beaf095e62320f0b4
#owner: https://api.github.com/users/stefan2904

#!/bin/bash

# if [[ -d $PASSED ]]; then
#     echo "$PASSED is a directory. Let's convert it."
# elif [[ -f $PASSED ]]; then
#     echo "$PASSED is a file. Try md2org instead?"
#     exit 2
# else
#     echo "$PASSED is not valid?"
#     exit 1
# fi

cd $1

for f in *.md; do
    echo ""
    filename="${f%%.*}"
    echo "** $filename" | sed 's/_/ /'

    pandoc --from markdown --to org "$f" \
    | sed 's/\[toc\]//' \
    | sed 's/\.\.\/\.\.\/_resources/file:\.\.\/_resources/' \
    | sed 's/\.\.\/\.\.\/work/\.\.\/work/' \
    | sed 's/\.\.\/journals/\.\.\/work\/journals/' \
    | sed 's/^\* /*** /' \
    | sed 's/^\*\* /*** /' \
    | sed 's/   :/:/' \
    | sed 's/☒/\[x\]/'  \
    | sed 's/☐/\[ \]/' 


done