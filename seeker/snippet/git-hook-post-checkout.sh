#date: 2025-03-03T16:55:59Z
#url: https://api.github.com/gists/927bd15094de35cc9505cda27c00f1b7
#owner: https://api.github.com/users/f-honcharenko

#!/bin/bash

# Run the custom script whenever a branch is checked out
"/Users/$USER/Library/Application Support/Code/User/update-copilot-branch-instructions.sh"

# Check if the custom script executed successfully
if [ $? -eq 0 ]; then
    # If the script was successful, print the success message
    echo "Successfully updated Copilot commit message generation instructions with the current Git branch name."
else
    # If the script failed, print an error message
    echo "Failed to update Copilot commit message generation instructions."
fi
