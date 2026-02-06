#date: 2026-02-06T17:35:56Z
#url: https://api.github.com/gists/5755393354261db8b746b03c839b2bca
#owner: https://api.github.com/users/tanguyfalconnet

#!/bin/bash

# https://kubernetes.io/docs/reference/access-authn-authz/authentication/#input-and-output-formats
echo '{
  "apiVersion": "client.authentication.k8s.io/v1",
  "kind": "ExecCredential",
  "status": {
    "token": "**********"://'$1' }}"
  }
}' | pass-cli inject