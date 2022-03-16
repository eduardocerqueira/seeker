#date: 2022-03-16T16:55:15Z
#url: https://api.github.com/gists/b5b02944518df299f8af9f2cd1245c95
#owner: https://api.github.com/users/Haris487

#!/bin/bash

downloadTrashFolder="download_trash_folder"

mkdir $downloadTrashFolder || echo "can not create folder $downloadTrashFolder"

fileExtensions=("csv" "kml" "jpg" "jpeg" "numbers" "pptx" "app" "json")
for fileExtension in ${fileExtensions[@]};
    do mkdir $downloadTrashFolder/$fileExtension || echo "can not create folder $fileExtension"
    for filename in ls *.$fileExtension
        do mv "$filename" ./$downloadTrashFolder/$fileExtension/ || echo echo mv \"$filename\" ./$downloadTrashFolder/$fileExtension/
    done
done

return;