#date: 2023-12-15T16:52:18Z
#url: https://api.github.com/gists/e6bd918be86315132524fd7115a9f4c3
#owner: https://api.github.com/users/allanpetersoon

import requests

TOKEN = "**********"

def get_matching_products(operation_id):
  endpoint = f'https://api.withleaf.io/services/beta/api/products/matching/operations/{operation_id}'
  headers = {'Authorization': "**********"
  response = requests.get(endpoint, headers=headers)
  return response.json()

operationId = "fill_with_your_operation_id"
matching_products = get_matching_products(operationId)
matching_productsoducts