#date: 2022-01-28T17:00:16Z
#url: https://api.github.com/gists/a3437be5adb89eae6d56a2565ad9ad51
#owner: https://api.github.com/users/Utopiah

#!/bin/bash
#
# opkg requirements :
# opkg install qpdf jq
# optionally imagemagick to generate the white page
# convert -size 1024x1448 xc:white /tmp/blank.pdf
#
# assumes ~/xochitl-data links to /home/root/.local/share/remarkable/xochitl/
#
# need to be able to renumber for each page added
# also need to update the metadata accordingly

LF="newpagebefore newpageafter addgrid movepage" #following by number e.g movepage20

hashFromTitle () { cd ~/.local/share/remarkable/xochitl/ && grep -l -i $1 *metadata | sed 's/.metadata//'; }

findActionPages() {
    hash=$(hashFromTitle $1)
    cd ~/xochitl-data;
    for b in $(grep -l -o 'ipage' $hash/*.json | sed 's/.*\/\(.*\)-metadata.*/\1/'); do
        echo $b
    done;
}

getPageNumberFromUUID(){
    cat $1.content|jq '.pages | index("'$2'") '
}

BOOK=$1
# opkg install imagemagick
# qpdf $BOOK.pdf --show-npages
# convert -size 1024x1448 xc:white /tmp/blank.pdf

cd ~/xochitl-data
UUIDNP=$(findActionPages $BOOK)
ZIPAGE=$(getPageNumberFromUUID $BOOK $UUIDNP)
sed -i "s/ipage/pageinserted/" $BOOK/$UUIDNP*.json
[ $ZIPAGE == "null" ] && exit
NEWPAGE=$(($ZIPAGE+1)) # PDF indexing starts at 1, not 0
echo inserting page $NEWPAGE
PAGES=$(jq -r .pageCount $BOOK.content)

mkdir -p tmp$BOOK
rm tmp$BOOK/page-*
cd tmp$BOOK

qpdf ../$BOOK.pdf --split-pages page
PAD=$( echo $PAGES|tr -d '\n'|wc -c )
for num in $(seq $PAGES -1 $NEWPAGE)
do
  P=$(printf "%0${PAD}d" $num )
  N=$(printf "%0${PAD}d" $(($num+1)) )
  [ $P == "0" ] && continue
  #fails for page 0 which doesn't exist
  mv page-$P page-$N
done

cp /tmp/blank.pdf page-$(printf "%0${PAD}d" $NEWPAGE )
qpdf --empty --pages page-* -- /tmp/new.pdf
mv /tmp/new.pdf ~/xochitl-data/$BOOK.pdf

cd $BOOK.highlights
mkdir -p previous
PUUIDS=$(jq -r '.pages['$ZIPAGE':-1]|reverse[]' ../$BOOK.content) # we insert a page, otherwise we keep on shifting the same one
# PUUIDS=$(jq -r '.pages['$NEWPAGE':-1][]' ../$BOOK.content) # we delete a page

# lose the last page if it has annotations/highlights as we don't have the upcoming UUID
# xochtil will only define it during the next document opening
# could rely on monitoring $BOOK.lock or $BOOK.content modification time newer than $BOOK.pdf after modification
# then check the new UUID as $BOOK.content | jq .pages[-1]

for PUUID in $PUUIDS; do
    if [ -f $PUUID.json ]; then
        mv $PUUID.json previous/
        if [ $PUUID != $(jq -r .pages[-1] ../$BOOK.content) ]; then
            NUUID=$(cat ../$BOOK.content|jq -r '.pages['$(cat ../$BOOK.content|jq '.pages | index("'$PUUID'") + 1')']')
            mv previous/$PUUID.json $NUUID.json
        fi # last page is saved on disk yet unusable
    fi
done
exit

cd ../$BOOK
mkdir -p previous
PUUIDS=$(ls *rm|sed "s/.rm//")
mv *.rm *-metadata.json previous
for PUUID in $PUUIDS; do
    NUUID=$(cat ../$BOOK.content|jq -r '.pages['$(cat ../$BOOK.content|jq '.pages | index("'$PUUID'") + 1')']')
    mv previous/$PUUID.rm $NUUID.rm
    mv previous/$PUUID-metadata.json $NUUID-metadata.json
done


# should also remove the layers functions that have been processed
# assuming 1 pass for now which is ... adding page 1
