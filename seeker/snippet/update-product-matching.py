#date: 2023-12-15T17:01:14Z
#url: https://api.github.com/gists/e13adfd7fff934faf347e95170e1dc1b
#owner: https://api.github.com/users/allanpetersoon

import requests

payload = {"status": "VALIDATED"}
or
payload = {"productId": "fill_with_expected_product_id"}

def updated_product_matches(operation_id, match_id, payload):
  endpoint = f'https://api.withleaf.io/services/beta/api/products/matching/operations/{operation_id}/matches/{match_id}'
  headers = {'Authorization': "**********"
  data = payload
  response = requests.put(endpoint, headers=headers, json=data)
  return response.json()

operationId = "fill_with_operation_id"
matchId = "fill_with_match_id"
print(updated_product_matches(operationId, matchId, payload))load))