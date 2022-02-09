#date: 2022-02-09T17:00:32Z
#url: https://api.github.com/gists/e2edfe02d6621fc9e89a10875832bd2b
#owner: https://api.github.com/users/JEPooleyOS

from os import environ
from osdatahub import Extent, FeaturesAPI

# Get OS Data Hub API key
key = environ.get("OS_API_KEY")

# Define extent
extent = Extent.from_ons_code("E09000001")

# Define product
product = "zoomstack_local_buildings"

# Query Features API
features_api = FeaturesAPI(key, product, extent)
local_buildings = features_api.query(limit=1000000)