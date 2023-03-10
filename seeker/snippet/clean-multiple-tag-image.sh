#date: 2023-03-10T17:09:23Z
#url: https://api.github.com/gists/50ef7ee610fb6643f7c5fb51e4b95dcf
#owner: https://api.github.com/users/haigeek

#!/bin/bash

# Define the age limit in seconds for images to be considered for deletion
AGE_LIMIT=$((10*24*60*60))
KEYWORDS=("192.168.1.181" "registry.tuyuansu.com.cn")


# Get the list of all Docker images on the system
IMAGE_LIST=$(docker image ls --format "{{.Repository}}:{{.Tag}}")

# Loop through each Docker image and check if it has multiple tags and meets the conditions for deletion
for image in $IMAGE_LIST
do
 # Check if the Docker image name contains any of the keywords
  if [[ " ${KEYWORDS[@]} " =~ " $(echo "$image" | cut -d':' -f1 | cut -d'/' -f2) " ]]
  then
    # Get the creation date of the Docker image as a UNIX timestamp
    CREATED_AT=$(docker inspect --format="{{.Created}}" $image | xargs -I{} date -d{} +%s)
    
    # Check if the Docker image meets the conditions for deletion
    if [[ $(($(date +%s)-$CREATED_AT)) -gt $AGE_LIMIT ]]
    then
      # Get the list of tags for the Docker image
      TAG_LIST=$(docker image inspect --format='{{.RepoTags}}' $image | sed 's/[][]//g' | sed 's/,/\n/g')
    
      # Check if the Docker image has multiple tags
      if [[ $(echo "$TAG_LIST" | wc -w) -gt 1 ]]
      then
        # Print the Docker image and its tags
        echo "Image: $image"
        echo "Tags: $TAG_LIST"
        docker image rm $image
        
        # Ask the user if they want to delete the Docker image and its tags
        # read -p "Do you want to delete this image and its tags? (y/n) " choice
        
        # Delete the Docker image and its tags if the user confirms
        # if [[ $choice == "y" ]]
        # then
          # docker image rm --force $image
        # fi
      fi
    fi
  fi
done
