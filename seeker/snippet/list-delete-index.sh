#date: 2024-09-20T17:08:58Z
#url: https://api.github.com/gists/ba6c3843858f022b739834dde1bf6e42
#owner: https://api.github.com/users/duboc

#!/bin/bash

# Get the access token
ACCESS_TOKEN= "**********"

# Define the base URL
BASE_URL="https://warehouse-visionai.googleapis.com/v1/projects/713488125678/locations/us-central1/corpora/4299188317952260006/indexes"

# List all indexes
echo "Listing indexes..."
curl -X GET \
     -H "Authorization: "**********"
     "$BASE_URL"

# Parse the JSON response to extract index IDs
echo "Deleting indexes..."
indexes=$(curl -X GET \
     -H "Authorization: "**********"
     "$BASE_URL" | jq -r '.indexes[].name' | sed 's/.*\/indexes\///')

# Delete each index
for index_id in $indexes; do
  curl -X DELETE \
       -H "Authorization: "**********"
       "$BASE_URL/$index_id"
done

echo "All indexes deleted."
ASE_URL/$index_id"
done

echo "All indexes deleted."
