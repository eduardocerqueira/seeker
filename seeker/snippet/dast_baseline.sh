#date: 2021-12-23T16:41:41Z
#url: https://api.github.com/gists/29c43bfb85d12e74e8c9f24bff2b85a2
#owner: https://api.github.com/users/soostech

#!/bin/sh

docker pull soosio/dast

docker run -it --rm soosio/dast --clientId=${SOOS_CLIENT_ID} --apiKey=${SOOS_API_KEY} --projectName="Project Name" Target_URL