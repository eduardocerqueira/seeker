#date: 2023-12-15T17:10:33Z
#url: https://api.github.com/gists/bb64252443bfa58725f61492ef87d72f
#owner: https://api.github.com/users/allanpetersoon

import requests

def get_a_product(product_id):
  endpoint = f'https://api.withleaf.io/services/beta/api/products/{product_id}'
  headers = {'Authorization': "**********"
  response = requests.get(endpoint, headers=headers)
  return response.json()

get_a_product("product_id")t_id")