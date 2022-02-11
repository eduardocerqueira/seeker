#date: 2022-02-11T16:42:41Z
#url: https://api.github.com/gists/01ab8d260c4b2edad0916f4f9eaff676
#owner: https://api.github.com/users/NemesisX1

#!/bin/bash

# We retrieve  all remote branches avalaible
values=$(git branch -r)

# Split them everytime we find a space character ' ' to get an array
values_array=(${values//' '/ })

# Then get the length of the array
length=${#values_array[@]}

# We loop through the array from the 3rd position
# because the first one are often Orign, --> and master
for (( i=3; i<${length}; i++ ));
do  
    # A simple control to avoid master branch if it still exist
    if [[ 'master' != ${values_array[$i]} ]];
    then
        echo 'Deleting '${values_array[$i]}'...'
        
        # We remove all the `origin/` inside array element
        value_without_origin=$(echo ${values_array[$i]//origin\/})

        #We delete remotely all the undesired branches
        git push --delete origin $value_without_origin
        
        echo 'Done !'
    fi
done