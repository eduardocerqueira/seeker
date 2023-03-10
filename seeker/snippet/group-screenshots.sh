#date: 2023-03-10T16:58:51Z
#url: https://api.github.com/gists/f257d68f7f14a3e39e299e648dfbbbd0
#owner: https://api.github.com/users/JamieW2

#################
# group-screenshots.sh
#   Groups a MacOS screenshots directory files
#   into folders by date and bursts of activity.
#   Secondary screenshots or grouped separately.
#
#  This undo command will flatten the directories
#    find . -mindepth 2 -type f -exec mv -i '{}' . ';'
##################

# don't break on spaces
OLDIFS=$IFS
IFS=$'\n'

# Separate secondary screen screenshots (usually just delete)
[[ -d secondary ]] || mkdir secondary
mv *"(2).png" secondary/

# Group files into bursts of activity
inactiveMinutes=20
lastDate=0
for f in $(ls *.png); do
 dateDiffMinutes=$(( ($(date -r $f +%s) - $lastDate) / 60 ))
 if [[ $dateDiffMinutes -gt $inactiveMinutes  ]]; then
  dir=$(date -r $f "+%Y-%m-%dT%H-%M-00")
  [[ -d $dir ]] || mkdir $dir
 fi
 lastDate=$(date -r "$f" +%s)
 [[ -f "$dir/$f" ]] || mv "$f" "$dir/$f"
done

# Merge directories with less than min files
minFiles=3
lastDir=
for d in */; do
 fileCount=$(ls $d | wc -l)
 if [[ $fileCount -gt $minFiles ]]; then
  lastDir=
 else
  if [[ -z "$lastDir" ]]; then
   lastDir="$d"
  fi
 fi
 if [[ $fileCount -le $minFiles ]]; then
  mv $d* $lastDir
  rm $d
 fi 
done

# reset IFS
IFS=$OLDIFS
