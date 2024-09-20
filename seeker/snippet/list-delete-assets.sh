#date: 2024-09-20T17:10:32Z
#url: https://api.github.com/gists/3d3ceac4b690daf2ea8391b875e496b4
#owner: https://api.github.com/users/duboc

#!/bin/bash

# Get the access token
ACCESS_TOKEN= "**********"

# Define the base URL
BASE_URL="https://warehouse-visionai.googleapis.com/v1/projects/713488125678/locations/us-central1/corpora/4299188317952260006/assets"

# List all assets
echo "Listing assets..."
curl -X GET \
     -H "Authorization: "**********"
     "$BASE_URL"

# Parse the JSON response to extract asset IDs
echo "Deleting assets..."
assets=$(curl -X GET \
     -H "Authorization: "**********"
     "$BASE_URL" | jq -r '.assets[].name' | sed 's/.*\/assets\///')

# Delete each asset
for asset_id in $assets; do
  curl -X DELETE \
       -H "Authorization: "**********"
       "$BASE_URL/$asset_id"
done

echo "All assets deleted."
BASE_URL/$asset_id"
done

echo "All assets deleted."
