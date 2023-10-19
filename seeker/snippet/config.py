#date: 2023-10-19T16:59:39Z
#url: https://api.github.com/gists/31642e621c0c9ebf3c45888442a142e2
#owner: https://api.github.com/users/miguelfferraz

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    DOMAIN = os.getenv("DOMAIN")
