#date: 2024-01-22T17:09:29Z
#url: https://api.github.com/gists/41e3474cd38ea431ec4b208748deb6cf
#owner: https://api.github.com/users/jmrada14

#!/bin/sh

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
  # Ask for OPENAI_API_KEY if not set
  echo "OPENAI_API_KEY is not set as an environment variable."
  printf "Please enter your OPENAI_API_KEY: "
  read OPENAI_API_KEY
  export OPENAI_API_KEY
fi

# Ask for a prompt to generate an image using DALL·E 3 model
printf "Enter a prompt to generate an image using DALL·E 3 model: "
read PROMPT

# Inform the user that the response is being generated
echo "Generating response..."

# Make the API call to generate an image with DALL·E 3
RESPONSE=$(curl -s -X POST "https://api.openai.com/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{
    \"model\": \"dall-e-3\",
    \"prompt\": \"$PROMPT\",
    \"n\": 1,
    \"size\": \"1024x1024\"
  }")

echo "Response from DALL·E 3 API:"
echo "$RESPONSE"