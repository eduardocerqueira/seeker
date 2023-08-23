#date: 2023-08-23T16:52:47Z
#url: https://api.github.com/gists/6fbf0d26ea5802a4d7c6135e142dd166
#owner: https://api.github.com/users/lancejohnson

# Instantiate the client with an endpoint.
endpoint="https://renovationreserve.myshopify.com/admin/api/2023-07/graphql.json"

# Provide a GraphQL query
query = """
query ProductsBySku($skuFilter: String!){
        products(first: 1, query: $skuFilter) {
          edges {
              node {
                  id
                  title
                  priceRangeV2 {
                      maxVariantPrice {
                          amount
                          currencyCode
                      }
                      minVariantPrice {
                          amount
                          currencyCode
                      }
                  }
                  vendor
                  variants(first: 10) {
                    edges {
                        node {
                            availableForSale
                            id
                            inventoryItem {
                                unitCost {
                                    amount
                                    currencyCode
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

skus = [
    "HP21-Black",
    "HP21-Black-SS"
]

# Prepare items for the request
variables = {"skuFilter": f"sku:{sku}"}

data = {
    "query": query,
    "variables": variables
}

headers = {
  'Content-Type': 'application/json',
  'X-Shopify-Access-Token': "**********"
}

resp = requests.post(url=endpoint, json=data, headers=headers)rs)