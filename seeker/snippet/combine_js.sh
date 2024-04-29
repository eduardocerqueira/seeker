#date: 2024-04-29T16:48:58Z
#url: https://api.github.com/gists/820eb6313d6cd619298c5352413a105b
#owner: https://api.github.com/users/archatas

#!/usr/bin/env bash

CURRENT_DIR=$(dirname "$0")
BASE_DIR=$CURRENT_DIR/../../..

# Define output filename
OUTPUT_FILE="$BASE_DIR/site_static/site/js/combined.js"

echo "" > $OUTPUT_FILE

# Fetch all external libraries
curl -s "https://code.jquery.com/jquery-2.2.4.js" >> $OUTPUT_FILE
curl -s "https://code.jquery.com/jquery-migrate-1.4.1.min.js" >> $OUTPUT_FILE
curl -s "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" >> $OUTPUT_FILE
# Append local libraries
cat "$BASE_DIR/site_static/site/js/jquery.mobile.custom.min.js" >> $OUTPUT_FILE
cat "$BASE_DIR/site_static/site/js/general.js" >> $OUTPUT_FILE

echo "JS files concatenated into: $OUTPUT_FILE"
