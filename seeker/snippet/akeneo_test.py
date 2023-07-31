#date: 2023-07-31T16:47:58Z
#url: https://api.github.com/gists/bdf852d2e73f8fa8ebe2320f45efa787
#owner: https://api.github.com/users/hildickethan-S73

import json
import requests

product = {
    "identifier": "123123",
    "attribute": "test_image",
    "scope": None,
    "locale": None,
}
path = "/home/x/image.jpeg" # image path
file_data = (
    "test.jpeg", # any new filename we choose
    open(path, "rb").read(),
    "image/jpeg", # file_data tuple must have the Content-Type
)
requests.post(
    "https://domain.com/api/rest/v1/media-files",
    headers={"Authorization": "**********"
    # (None, ) mandatory for no filename in the request, "" filename doesn't work
    # mandatory json.dumps to stringify
    files={"product": (None, json.dumps(product)), "file": file_data},
)ta},
)