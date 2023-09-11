#date: 2023-09-11T16:55:54Z
#url: https://api.github.com/gists/b07b8bfd7ab102c7a4cd423c2b929378
#owner: https://api.github.com/users/GuyPaddock

#!/bin/bash

# Define the Pantheon upstream ID
UPSTREAM_ID="21e1fada-199c-492b-97bd-0b36b53a9da0"

# Authenticate with Terminus (replace with your authentication method)
# terminus auth:login

# Initialize the CSV output with headers
echo "Site Name,Install Profile Version"

# Get a list of non-frozen Drupal 7 sites based on the specified upstream
SITES=$(terminus site:list --upstream "$UPSTREAM_ID" --fields=name --format=csv --filter="frozen=" | tail -n +2)

# Define the target environment
ENVIRONMENT="dev" # Change to your desired environment (e.g., test, live)

# Loop through the sites and check the install profile version
for SITE_NAME in $SITES; do
  SITE="$SITE_NAME.$ENVIRONMENT"

  # Use Terminus to execute rq and store the JSON output
  REQUIREMENTS_JSON=$(terminus drush $SITE -- rq --format=json)

  # Extract the install profile value from the JSON
  INSTALL_PROFILE_VALUE=$(echo $REQUIREMENTS_JSON | jq -r '.install_profile.value')

  # Extract the version from the install profile value using a more specific regex
  INSTALL_PROFILE_VERSION=$(echo $INSTALL_PROFILE_VALUE | sed -n 's/.*>\(.*\)<\/em>.*/\1/p')

  # Output the results in CSV format to standard output
  echo "$SITE_NAME,$INSTALL_PROFILE_VERSION"
done