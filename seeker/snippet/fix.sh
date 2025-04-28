#date: 2025-04-28T17:07:37Z
#url: https://api.github.com/gists/18e621782c3037b5e3a1cd7de8c1101f
#owner: https://api.github.com/users/Dpbm

#!/bin/bash

i=0
image_i=0

IMAGES_FOLDER="../images"

mkdir -p "$IMAGES_FOLDER"

TMP_HTML="../tmp.html"
TMP_MDX="tmp.mdx"

for file in *.html; do
    title=$(
        echo "$file" | 
        sed 's/\.html//g' | 
        sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2}_//g' | 
        sed -E 's/Ci-ncia(s)?-da-[cC]omputa[c-]-o(-dia)?-/Ciências da computação dia /g' |  
        sed -E 's/-([2-9a-z]{13}|[0-9a-z]{12}|[0-9a-z]{11})//g' | 
        sed 's/Links--teis-para-quem-quer-criar-um-sistema/Links uteis para quem quer criar um sistema operacional/g'
        )

    echo "cleaning: ${title}..."

    folder_name="folder_$i"

    mkdir -p "../$folder_name"

    # clear attributes and some useless tags
    cat $file | 
        sed -E 's/(class|name|id|rel|data-href|data-field|data-thumbnail-img-id|target|data-media-id|style|data-media-style|title|data-image-id|data-height|data-width|data-is-featured)="[^"]*"//g' | 
        sed -E '/<header>/,/<\/header>/g' | 
        sed -E '/<footer>/,/<\/footer>/g' | 
        awk 'BEGIN{inside=0;removed=0} !removed && /<section[[:space:]]*>/ {inside=1} !removed && inside { if (/<\/section>/) {inside=0;removed=1}; next } {print}' > "$TMP_HTML"

    # get images
    images=$(
        cat "$TMP_HTML" |
        grep -Eo '(https?|ftp)://[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9./?=#&_%+-\*]*)?(png|jpg|jpeg|gif)'
    )

    # download and map src to local images
    for image_url in $images; do
        image_name="image_$image_i.png"
        image_path="$IMAGES_FOLDER/$image_name"

        echo "Downloading image: $image_url"
        echo "Saving image at: $image_path"
        wget -O "$image_path" "$image_url" 

        fixed_url=$(echo "$image_url" | sed 's/\*/\\*/g')
        echo "fixed url: $fixed_url"
        echo "Replacing html cdn url by local image..."
        sed -i "s|$fixed_url|$image_name|g" "$TMP_HTML"

        image_i=$(( $image_i + 1 ))

        sleep 0.3
    done

    target_mdx="../$folder_name/post.mdx"
    pandoc "$TMP_HTML" -f html -t markdown -o "$TMP_MDX"

    cat "$TMP_MDX" | 
        awk '!/:::/ && !/<div>/ && !/<\/div>/ && !/------------------------------------------------------------------------/' |
        sed '0,/###/{/###/d}' > "$target_mdx"

    subtitle=$(awk 'NF {print; exit}' "$target_mdx")
    sed -i "0,/$subtitle/{/$subtitle/d}" "$target_mdx"

    rm -rf "$TMP_HTML"

    mv "../$folder_name" "../../personal-blog/app/content/"
    cd "../../personal-blog/"
    node ./add_post_build/posts/addPost.js "$title" "$subtitle" "./app/content/$folder_name"
    cd "../blog-posts-fix/posts"


    i=$(($i + 1))
done

mv "$IMAGES_FOLDER/*.png" "../../personal-blog/app/public/"
