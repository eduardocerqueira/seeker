#date: 2024-03-13T17:01:57Z
#url: https://api.github.com/gists/26cc5537174070177c51d82c1376817a
#owner: https://api.github.com/users/jotpalch

#!/bin/bash

SERVICES=("copilot-gpt4-service" "chatgpt-next-web")
PORTS=(30001 30002)
API_KEY="MASKED_FOR_SECURITY"
CODE="MASKED_FOR_SECURITY"
BASE_URL="MASKED_FOR_SECURITY"

# Remember to run `chmod +x run.sh` before running the script

for SERVICE in "${SERVICES[@]}"; do
    docker stop $SERVICE
    docker rm $SERVICE
done

docker run -d \
    --name ${SERVICES[0]} \
    --restart always   \
    -p ${PORTS[0]}:8080   \
    -e HOST=0.0.0.0   \
    aaamoon/${SERVICES[0]}:latest

docker run -d \
    --name ${SERVICES[1]} \
    -p ${PORTS[1]}:3000 \
    -e OPENAI_API_KEY=$API_KEY \
    -e CODE=$CODE  \
    -e BASE_URL=${BASE_URL}:${PORTS[0]}  \
    yidadaa/${SERVICES[1]}
