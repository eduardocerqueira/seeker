#date: 2023-07-07T17:10:27Z
#url: https://api.github.com/gists/c76f713ac75464b798a7c2e3398ccba5
#owner: https://api.github.com/users/rolfen

#/bin/bash
echo “Enter source”
read dir
 
echo “Enter destination”
read dest
 
for trgt in $(cd $dir && find . -type f \( -iname \*.ARW -o -iname \*.ORF \) );
do
	echo "$dir/$trgt >> $dest/$trgt.JPG" 
	echo "mkdir "`dirname $dest/$trgt`
	exiftool  -m $dir/$trgt  -b -previewimage -ext ORF -ext ARW > $dest/$trgt.JPG 
	exiftool  -overwrite_original -m -tagsfromfile $dir/$trgt "-all:all>all:all" $dest/$trgt.JPG
done;