#date: 2025-05-28T16:48:47Z
#url: https://api.github.com/gists/a566aacbdc516a20d3cd6f4ad62ec158
#owner: https://api.github.com/users/jaehoo

# Consume published iflow service
# =========================================
export CLIENT_ID='<CLIENT_ID>'
export CLIENT_SECRET= "**********"
export TOKEN_ENDPOINT= "**********"
export WS_ENDPOINT=<IFLOW_ENDPOINT>

printf "\n endpoint: $WS_ENDPOINT \n"

response= "**********"
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id= "**********"=$CLIENT_SECRET&grant_type=client_credentials")

ACCESS_TOKEN=$(echo "$response" | grep -oE '"access_token": "**********":"([^"]+)"/\1/')

printf "\n\nRequest Add ...\n"
curl -X POST -w "\n%{http_code}" "$WS_ENDPOINT" \
  -H "Authorization: "**********"
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Add\"" \
  -d @add.xml

printf "\n\nRequest Subtract ...\n"
curl -X POST -w "\n%{http_code}" "$WS_ENDPOINT" \
  -H "Authorization: "**********"
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Subtract\"" \
  -d @subtract.xml

printf "\n\nRequest Multiply ...\n"
curl -X POST -w "\n%{http_code}" "$WS_ENDPOINT" \
  -H "Authorization: "**********"
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Multiply\"" \
  -d @multiply.xml

printf "\n\nRequest Divide ...\n"
curl -X POST -w "\n%{http_code}" "$WS_ENDPOINT" \
  -H "Authorization: "**********"
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Divide\"" \
  -d @divide.xml

printf "\n\nRequest Unsupported operation...\n"
 curl -X POST -w "\n%{http_code}" "$WS_ENDPOINT" \
  -H "Authorization: "**********"
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Unsupported\"" \
  -d @unsupported.xmlrer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "SOAPAction: \"Unsupported\"" \
  -d @unsupported.xml