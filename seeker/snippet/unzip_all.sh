#date: 2025-03-21T17:01:09Z
#url: https://api.github.com/gists/07ba5c65d66cd50a6ac1a90800e72584
#owner: https://api.github.com/users/enn-dee

#!/bin/bash

# make sure to place it in dir in which want to unzip folders & give this file exec permissions
for zipFile in *.zip; do
   
    folderName="${zipFile%.zip}"
    
   
    mkdir "$folderName"
    

    unzip "$zipFile" -d "$folderName"
    
    echo "Unzipped $zipFile into $folderName/"
done
