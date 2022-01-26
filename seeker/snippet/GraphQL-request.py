#date: 2022-01-26T17:02:31Z
#url: https://api.github.com/gists/6e0857fde811aa870ddf963c31151c4d
#owner: https://api.github.com/users/tchak

import requests

token = "..."

dossierId = "..."
instructeurId = "..."
motivation = "..."

query = """mutation dossierAccepter($input: DossierAccepterInput!) {
  dossierAccepter(input: $input) {
    dossier {
      id
    }
    errors {
      message
    }
  }
}"""

variables = {
  'input': {
    'dossierId': dossierId,
    'instructeurId': instructeurId,
    'motivation': motivation
  }
}

headers = {
  'authorization': 'Bearer %s' % token,
  'content-type': 'application/json'
}

body = {
  'operationName': 'dossierAccepter',
  'query': query,
  'variables': variables
}

response = requests.post(
  'https://www.demarches-simplifiees.fr/api/v2/graphql',
  headers=headers
  json=body
)