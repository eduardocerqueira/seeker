#date: 2022-07-13T17:15:19Z
#url: https://api.github.com/gists/0751b1179c92279ce88ac1eead80772e
#owner: https://api.github.com/users/dansteren

#!/bin/bash

# For a given directory, navigate into the dfx_generated folder and loop through
# all agent folders, and comment out the last line in the index.js file.
cd dfx_generated || exit
for directory in */; do
    echo "Processing $directory"
    STORED_LAST_LINE=$(tail --lines=1 "$directory"/index.js)
    echo "The saved text is: $STORED_LAST_LINE"
    sed -i '$ d' "$directory"/index.js
    echo "// $STORED_LAST_LINE" >> "$directory"/index.js
done
