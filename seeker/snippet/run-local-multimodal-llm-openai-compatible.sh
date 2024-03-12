#date: 2024-03-12T16:43:57Z
#url: https://api.github.com/gists/acaa476865821b02813b8a8e88e59c13
#owner: https://api.github.com/users/cagataycali

#!/bin/bash

# Define the file path
FILE="llava-v1.5-7b-q4.llamafile"

# Check if the file exists
if [ -f "$FILE" ]; then
    echo "$FILE exists. Starting."
else
    echo "$FILE does not exist, downloading now."

    # Download the LLaMA file from Hugging Face
    wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4.llamafile?download=true -O $FILE
fi

# Make the downloaded file executable
chmod +x $FILE;

# Kill the LLaMA process if it is already running
killall $FILE;

# Start the LLaMA process with the specified option in the background
./$FILE -ngl 9999 &

open http://localhost:8080;