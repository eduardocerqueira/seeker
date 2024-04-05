#date: 2024-04-05T16:57:38Z
#url: https://api.github.com/gists/1a104bab17a9818ce05be195e8281f36
#owner: https://api.github.com/users/ansgomez

#!/bin/bash 
# Most vector images can be extracted as EPS from the command line using pdftocairo + sed.
# https://superuser.com/questions/302045/how-to-extract-vectors-from-a-pdf-file/884445

# However, there was one file (created by LaTeX) that failed to extract because pdftocairo 
# was rasterizing it into a bitmap, so I created a script that uses xml_grep to extract the
# image as an SVG file instead.

# b9 2019

if [[ -z "$1" ]]; then
    echo "Usage: $0 <filename.pdf> [pagenumber]"
    exit 1
fi

file="$1"
page="$2"

if [ "$page" ]; then
    pageopt="-f $page -l $page"
fi

pdftocairo "$file"  $pageopt  -svg temp.svg

for surface in $(grep -o '<g id="surface[^"]*"' temp.svg | grep -o '".*"' | tr -d '"'); do

    if [[ "$surface" == "surface1" ]]; then
	#echo "Skipping surface1 which is entire page"
	continue
    fi
    
    output="${file%.*}-$surface.svg"
    echo "Extracting: $output"

    (
	echo "<svg>" 
	xml_grep --nowrap --pretty indented '//defs' temp.svg
	xml_grep --nowrap --pretty indented 'g[@id="'"$surface"'"]' temp.svg
	echo "</svg>"
    )  > $output

    # Make it a valid SVG file with a proper bounding box
    # You may need to `apt install librsvg2-bin` for this to work.
    rsvg-convert $output -f svg -o temp.svg  &&  mv temp.svg $output

done


if [[ -z "$output" ]]; then
    echo "No additional surfaces found in this PDF."
fi

rm -f temp.svg
