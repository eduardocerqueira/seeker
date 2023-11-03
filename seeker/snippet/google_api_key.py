#date: 2023-11-03T17:02:48Z
#url: https://api.github.com/gists/658f0911766c88bcd690d5b70bae66c8
#owner: https://api.github.com/users/bilgeyucel

import os
from google.colab import userdata

openai_api_key = os.getenv("OPENAI_API_KEY", userdata.get("OPENAI_API_KEY")) 