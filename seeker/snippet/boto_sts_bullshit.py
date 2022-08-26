#date: 2022-08-26T17:07:28Z
#url: https://api.github.com/gists/b12a4ecad92eb8f71d2855e278efd791
#owner: https://api.github.com/users/mikegrima

# If doing STS things, you will need to make sure that you use the proper STS endpoint now.
# You need to remember that you need to pass in the endpoint URL. Use this for CloudAux:
from typing import Any, Dict, List

from botocore.client import BaseClient
from cloudaux import sts_conn
from cloudaux.aws.decorators import paginated

ROLE_TO_ASSUME = "AssumeThisRole"
ACCOUNT_NUMBER = "012345678910"
REGION = "af-south-1"

@paginated("Keys", request_pagination_marker="Marker", response_pagination_marker="NextMarker")
@sts_conn("kms")
def list_keys(client: BaseClient = None, **kwargs) -> List[Dict[str, Any]]:
    return client.list_keys(**kwargs)

# Using it:
kms_keys = list_keys(
  account_number=ACCOUNT_NUMBER,
  assume_role=ROLE_TO_ASSUME,
  region=REGION,
  sts_client_kwargs={"endpoint_url": f"https://sts.{REGION}.amazonaws.com", "region_name": REGION},
)
