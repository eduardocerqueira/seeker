#date: 2025-05-28T16:45:47Z
#url: https://api.github.com/gists/e1170552e5a97a87cc9af3be5c2016ed
#owner: https://api.github.com/users/jaehoo

#!/bin/bash

# Consume local soap service
# =========================================

export ENDPOINT="http://localhost:8080/ws"

curl -X POST $ENDPOINT \
  -H "Content-Type: text/xml;charset=UTF-8" \
  -H "SOAPAction: \"Add\"" \
  -d @add.xml

curl -X POST $ENDPOINT \
  -H "Content-Type: text/xml;charset=UTF-8" \
  -H "SOAPAction: \"Subtract\"" \
  -d @subtract.xml

curl -X POST $ENDPOINT \
  -H "Content-Type: text/xml;charset=UTF-8" \
  -H "SOAPAction: \"Multiply\"" \
  -d @multiply.xml

curl -X POST $ENDPOINT \
  -H "Content-Type: text/xml;charset=UTF-8" \
  -H "SOAPAction: \"Divide\"" \
  -d @divide.xml
